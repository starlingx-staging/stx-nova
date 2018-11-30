#    Copyright 2013 IBM Corp.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#
# Copyright (c) 2013-2017 Wind River Systems, Inc.
#

"""Handles database requests from other nova services."""

import contextlib
import copy
import functools

from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils
import six

from nova import availability_zones
from nova.compute import instance_actions
from nova.compute import rpcapi as compute_rpcapi
from nova.compute import task_states
from nova.compute import utils as compute_utils
from nova.compute.utils import wrap_instance_event
from nova.compute import vm_states
from nova.conductor.tasks import live_migrate
from nova.conductor.tasks import migrate
from nova import context as nova_context
from nova.db import base
from nova import exception
from nova.i18n import _
from nova import image
from nova import manager
from nova import network
from nova import notifications
from nova import objects
from nova.objects import base as nova_object
from nova import profiler
from nova import rpc
from nova.scheduler import client as scheduler_client
from nova.scheduler import utils as scheduler_utils
from nova import servicegroup
from nova import utils
from nova.volume import cinder

LOG = logging.getLogger(__name__)
CONF = cfg.CONF


def targets_cell(fn):
    """Wrap a method and automatically target the instance's cell.

    This decorates a method with signature func(self, context, instance, ...)
    and automatically targets the context with the instance's cell
    mapping. It does this by looking up the InstanceMapping.
    """
    @functools.wraps(fn)
    def wrapper(self, context, *args, **kwargs):
        instance = kwargs.get('instance') or args[0]
        try:
            im = objects.InstanceMapping.get_by_instance_uuid(
                context, instance.uuid)
        except exception.InstanceMappingNotFound:
            LOG.error('InstanceMapping not found, unable to target cell',
                      instance=instance)
            im = None
        else:
            LOG.debug('Targeting cell %(cell)s for conductor method %(meth)s',
                      {'cell': im.cell_mapping.identity,
                       'meth': fn.__name__})
            # NOTE(danms): Target our context to the cell for the rest of
            # this request, so that none of the subsequent code needs to
            # care about it.
            nova_context.set_target_cell(context, im.cell_mapping)
        return fn(self, context, *args, **kwargs)
    return wrapper


class ConductorManager(manager.Manager):
    """Mission: Conduct things.

    The methods in the base API for nova-conductor are various proxy operations
    performed on behalf of the nova-compute service running on compute nodes.
    Compute nodes are not allowed to directly access the database, so this set
    of methods allows them to get specific work done without locally accessing
    the database.

    The nova-conductor service also exposes an API in the 'compute_task'
    namespace.  See the ComputeTaskManager class for details.
    """

    target = messaging.Target(version='3.0')

    def __init__(self, *args, **kwargs):
        super(ConductorManager, self).__init__(service_name='conductor',
                                               *args, **kwargs)
        self.compute_task_mgr = ComputeTaskManager()
        self.additional_endpoints.append(self.compute_task_mgr)

    # NOTE(hanlind): This can be removed in version 4.0 of the RPC API
    def provider_fw_rule_get_all(self, context):
        # NOTE(hanlind): Simulate an empty db result for compat reasons.
        return []

    def _object_dispatch(self, target, method, args, kwargs):
        """Dispatch a call to an object method.

        This ensures that object methods get called and any exception
        that is raised gets wrapped in an ExpectedException for forwarding
        back to the caller (without spamming the conductor logs).
        """
        try:
            # NOTE(danms): Keep the getattr inside the try block since
            # a missing method is really a client problem
            return getattr(target, method)(*args, **kwargs)
        except Exception:
            raise messaging.ExpectedException()

    def object_class_action_versions(self, context, objname, objmethod,
                                     object_versions, args, kwargs):
        objclass = nova_object.NovaObject.obj_class_from_name(
            objname, object_versions[objname])
        args = tuple([context] + list(args))
        result = self._object_dispatch(objclass, objmethod, args, kwargs)
        # NOTE(danms): The RPC layer will convert to primitives for us,
        # but in this case, we need to honor the version the client is
        # asking for, so we do it before returning here.
        # NOTE(hanlind): Do not convert older than requested objects,
        # see bug #1596119.
        if isinstance(result, nova_object.NovaObject):
            target_version = object_versions[objname]
            requested_version = versionutils.convert_version_to_tuple(
                target_version)
            actual_version = versionutils.convert_version_to_tuple(
                result.VERSION)
            do_backport = requested_version < actual_version
            other_major_version = requested_version[0] != actual_version[0]
            if do_backport or other_major_version:
                result = result.obj_to_primitive(
                    target_version=target_version,
                    version_manifest=object_versions)
        return result

    def object_action(self, context, objinst, objmethod, args, kwargs):
        """Perform an action on an object."""
        oldobj = objinst.obj_clone()
        result = self._object_dispatch(objinst, objmethod, args, kwargs)
        updates = dict()
        # NOTE(danms): Diff the object with the one passed to us and
        # generate a list of changes to forward back
        for name, field in objinst.fields.items():
            if not objinst.obj_attr_is_set(name):
                # Avoid demand-loading anything
                continue
            if (not oldobj.obj_attr_is_set(name) or
                    getattr(oldobj, name) != getattr(objinst, name)):
                updates[name] = field.to_primitive(objinst, name,
                                                   getattr(objinst, name))
        # This is safe since a field named this would conflict with the
        # method anyway
        updates['obj_what_changed'] = objinst.obj_what_changed()
        return updates, result

    def object_backport_versions(self, context, objinst, object_versions):
        target = object_versions[objinst.obj_name()]
        LOG.debug('Backporting %(obj)s to %(ver)s with versions %(manifest)s',
                  {'obj': objinst.obj_name(),
                   'ver': target,
                   'manifest': ','.join(
                       ['%s=%s' % (name, ver)
                       for name, ver in object_versions.items()])})
        return objinst.obj_to_primitive(target_version=target,
                                        version_manifest=object_versions)

    def reset(self):
        objects.Service.clear_min_version_cache()


@contextlib.contextmanager
def try_target_cell(context, cell):
    """If cell is not None call func with context.target_cell.

    This is a method to help during the transition period. Currently
    various mappings may not exist if a deployment has not migrated to
    cellsv2. If there is no mapping call the func as normal, otherwise
    call it in a target_cell context.
    """
    if cell:
        with nova_context.target_cell(context, cell) as cell_context:
            yield cell_context
    else:
        yield context


@contextlib.contextmanager
def obj_target_cell(obj, cell):
    """Run with object's context set to a specific cell"""
    with try_target_cell(obj._context, cell) as target:
        with obj.obj_alternate_context(target):
            yield target


@profiler.trace_cls("rpc")
class ComputeTaskManager(base.Base):
    """Namespace for compute methods.

    This class presents an rpc API for nova-conductor under the 'compute_task'
    namespace.  The methods here are compute operations that are invoked
    by the API service.  These methods see the operation to completion, which
    may involve coordinating activities on multiple compute nodes.
    """

    target = messaging.Target(namespace='compute_task', version='1.17')

    def __init__(self):
        super(ComputeTaskManager, self).__init__()
        self.compute_rpcapi = compute_rpcapi.ComputeAPI()
        self.image_api = image.API()
        self.network_api = network.API()
        self.servicegroup_api = servicegroup.API()
        self.volume_api = cinder.API()
        self.scheduler_client = scheduler_client.SchedulerClient()
        self.notifier = rpc.get_notifier('compute', CONF.host)

    def reset(self):
        LOG.info('Reloading compute RPC API')
        compute_rpcapi.LAST_VERSION = None
        self.compute_rpcapi = compute_rpcapi.ComputeAPI()

    # TODO(tdurakov): remove `live` parameter here on compute task api RPC
    # version bump to 2.x
    @messaging.expected_exceptions(
        exception.NoValidHost,
        exception.ComputeServiceUnavailable,
        exception.ComputeHostNotFound,
        exception.InvalidHypervisorType,
        exception.InvalidCPUInfo,
        exception.UnableToMigrateToSelf,
        exception.DestinationHypervisorTooOld,
        exception.InvalidLocalStorage,
        exception.InvalidSharedStorage,
        exception.HypervisorUnavailable,
        exception.InstanceInvalidState,
        exception.MigrationPreCheckError,
        exception.MigrationPreCheckClientException,
        exception.LiveMigrationWithOldNovaNotSupported,
        exception.UnsupportedPolicyException)
    @targets_cell
    @wrap_instance_event(prefix='conductor')
    def migrate_server(self, context, instance, scheduler_hint, live, rebuild,
            flavor, block_migration, disk_over_commit, reservations=None,
            clean_shutdown=True, request_spec=None):
        if instance and not isinstance(instance, nova_object.NovaObject):
            # NOTE(danms): Until v2 of the RPC API, we need to tolerate
            # old-world instance objects here
            attrs = ['metadata', 'system_metadata', 'info_cache',
                     'security_groups']
            instance = objects.Instance._from_db_object(
                context, objects.Instance(), instance,
                expected_attrs=attrs)
        # NOTE: Remove this when we drop support for v1 of the RPC API
        if flavor and not isinstance(flavor, objects.Flavor):
            # Code downstream may expect extra_specs to be populated since it
            # is receiving an object, so lookup the flavor to ensure this.
            flavor = objects.Flavor.get_by_id(context, flavor['id'])
        if live and not rebuild and not flavor:
            self._live_migrate(context, instance, scheduler_hint,
                               block_migration, disk_over_commit, request_spec)
        elif not live and not rebuild and flavor:
            instance_uuid = instance.uuid
            with compute_utils.EventReporter(context, 'cold_migrate',
                                             instance_uuid):
                self._cold_migrate(context, instance, flavor,
                                   scheduler_hint['filter_properties'],
                                   reservations, clean_shutdown, request_spec)
        else:
            raise NotImplementedError()

    def _cold_migrate(self, context, instance, flavor, filter_properties,
                      reservations, clean_shutdown, request_spec):
        image = utils.get_image_from_system_metadata(
            instance.system_metadata)

        # NOTE(sbauza): If a reschedule occurs when prep_resize(), then
        # it only provides filter_properties legacy dict back to the
        # conductor with no RequestSpec part of the payload.
        if not request_spec:
            # Make sure we hydrate a new RequestSpec object with the new flavor
            # and not the nested one from the instance

            # WRS: these hints are needed by the vcpu filter
            hints = filter_properties.get('scheduler_hints', {})
            hints['task_state'] = instance.task_state or ""
            hints['host'] = instance.host or ""
            hints['node'] = instance.node or ""
            filter_properties['scheduler_hints'] = hints

            request_spec = objects.RequestSpec.from_components(
                context, instance.uuid, image,
                flavor, instance.numa_topology, instance.pci_requests,
                filter_properties, None, instance.availability_zone)
        else:
            # NOTE(sbauza): Resizes means new flavor, so we need to update the
            # original RequestSpec object for make sure the scheduler verifies
            # the right one and not the original flavor
            request_spec.flavor = flavor

            # WRS: these hints are needed by the vcpu filter
            hints = {}
            hints['task_state'] = [instance.task_state or ""]
            hints['host'] = [instance.host or ""]
            hints['node'] = [instance.node or ""]
            if request_spec.obj_attr_is_set('scheduler_hints') and \
                    request_spec.scheduler_hints:
                request_spec.scheduler_hints.update(hints)
            else:
                request_spec.scheduler_hints = hints

        task = self._build_cold_migrate_task(context, instance, flavor,
                                             request_spec,
                                             reservations, clean_shutdown)
        # TODO(sbauza): Provide directly the RequestSpec object once
        # _set_vm_state_and_notify() accepts it
        legacy_spec = request_spec.to_legacy_request_spec_dict()
        try:
            request_spec = task.execute()
        except exception.NoValidHost as ex:
            vm_state = instance.vm_state
            if not vm_state:
                vm_state = vm_states.ACTIVE
            updates = {'vm_state': vm_state, 'task_state': None}
            self._set_vm_state_and_notify(context, instance.uuid,
                                          'migrate_server',
                                          updates, ex, legacy_spec)

            # if the flavor IDs match, it's migrate; otherwise resize
            if flavor.id == instance.instance_type_id:
                msg = _("No valid host found for cold migrate")
            else:
                msg = _("No valid host found for resize")
            LOG.error("%s", msg)
            raise
        except exception.UnsupportedPolicyException as ex:
            with excutils.save_and_reraise_exception():
                vm_state = instance.vm_state
                if not vm_state:
                    vm_state = vm_states.ACTIVE
                updates = {'vm_state': vm_state, 'task_state': None}
                self._set_vm_state_and_notify(context, instance.uuid,
                                              'migrate_server',
                                              updates, ex, legacy_spec)
        except Exception as ex:
            with excutils.save_and_reraise_exception():
                updates = {'vm_state': instance.vm_state,
                           'task_state': None}
                self._set_vm_state_and_notify(context, instance.uuid,
                                              'migrate_server',
                                              updates, ex, legacy_spec)
        # NOTE(sbauza): Make sure we persist the new flavor in case we had
        # a successful scheduler call if and only if nothing bad happened
        if request_spec.obj_what_changed():
            request_spec.save()

    def _set_vm_state_and_notify(self, context, instance_uuid, method, updates,
                                 ex, request_spec):
        scheduler_utils.set_vm_state_and_notify(
                context, instance_uuid, 'compute_task', method, updates,
                ex, request_spec)

    def _cleanup_allocated_networks(
            self, context, instance, requested_networks):
        try:
            # If we were told not to allocate networks let's save ourselves
            # the trouble of calling the network API.
            if not (requested_networks and requested_networks.no_allocate):
                self.network_api.deallocate_for_instance(
                    context, instance, requested_networks=requested_networks)
        except Exception:
            LOG.exception('Failed to deallocate networks', instance=instance)
            return

        instance.system_metadata['network_allocated'] = 'False'
        try:
            instance.save()
        except exception.InstanceNotFound:
            # NOTE: It's possible that we're cleaning up the networks
            # because the instance was deleted.  If that's the case then this
            # exception will be raised by instance.save()
            pass

    @targets_cell
    @wrap_instance_event(prefix='conductor')
    def live_migrate_instance(self, context, instance, scheduler_hint,
                              block_migration, disk_over_commit, request_spec):
        self._live_migrate(context, instance, scheduler_hint,
                           block_migration, disk_over_commit, request_spec)

    def _live_migrate(self, context, instance, scheduler_hint,
                      block_migration, disk_over_commit, request_spec):
        destination = scheduler_hint.get("host")

        def _set_vm_state(context, instance, ex, vm_state=None,
                          task_state=None):
            request_spec = {'instance_properties': {
                'uuid': instance.uuid, },
            }
            scheduler_utils.set_vm_state_and_notify(context,
                instance.uuid,
                'compute_task', 'migrate_server',
                dict(vm_state=vm_state,
                     task_state=task_state,
                     expected_task_state=task_states.MIGRATING,),
                ex, request_spec)

        migration = objects.Migration(context=context.elevated())
        migration.dest_compute = destination
        migration.status = 'accepted'
        migration.instance_uuid = instance.uuid
        migration.source_compute = instance.host
        migration.source_node = instance.node
        migration.migration_type = 'live-migration'
        if instance.obj_attr_is_set('flavor'):
            migration.old_instance_type_id = instance.flavor.id
            migration.new_instance_type_id = instance.flavor.id
        else:
            migration.old_instance_type_id = instance.instance_type_id
            migration.new_instance_type_id = instance.instance_type_id
        migration.create()

        task = self._build_live_migrate_task(context, instance, destination,
                                             block_migration, disk_over_commit,
                                             migration, request_spec)
        try:
            task.execute()
        except (exception.NoValidHost,
                exception.ComputeHostNotFound,
                exception.ComputeServiceUnavailable,
                exception.InvalidHypervisorType,
                exception.InvalidCPUInfo,
                exception.UnableToMigrateToSelf,
                exception.DestinationHypervisorTooOld,
                exception.InvalidLocalStorage,
                exception.InvalidSharedStorage,
                exception.HypervisorUnavailable,
                exception.InstanceInvalidState,
                exception.MigrationPreCheckError,
                exception.MigrationPreCheckClientException,
                exception.LiveMigrationWithOldNovaNotSupported,
                exception.MigrationSchedulerRPCError) as ex:
            with excutils.save_and_reraise_exception():
                # TODO(johngarbutt) - eventually need instance actions here
                _set_vm_state(context, instance, ex, instance.vm_state)
                migration.status = 'error'
                migration.save()
        except Exception as ex:
            LOG.error('Migration of instance %(instance_id)s to host'
                      ' %(dest)s unexpectedly failed.',
                      {'instance_id': instance.uuid, 'dest': destination},
                      exc_info=True)
            # Reset the task state to None to indicate completion of
            # the operation as it is done in case of known exceptions.
            _set_vm_state(context, instance, ex, vm_states.ERROR,
                          task_state=None)
            migration.status = 'error'
            migration.save()
            raise exception.MigrationError(reason=six.text_type(ex))

    def _build_live_migrate_task(self, context, instance, destination,
                                 block_migration, disk_over_commit, migration,
                                 request_spec=None):
        return live_migrate.LiveMigrationTask(context, instance,
                                              destination, block_migration,
                                              disk_over_commit, migration,
                                              self.compute_rpcapi,
                                              self.servicegroup_api,
                                              self.scheduler_client,
                                              request_spec)

    def _build_cold_migrate_task(self, context, instance, flavor,
                                 request_spec, reservations,
                                 clean_shutdown):
        return migrate.MigrationTask(context, instance, flavor,
                                     request_spec,
                                     reservations, clean_shutdown,
                                     self.compute_rpcapi,
                                     self.scheduler_client)

    def _destroy_build_request(self, context, instance):
        # The BuildRequest needs to be stored until the instance is mapped to
        # an instance table. At that point it will never be used again and
        # should be deleted.
        build_request = objects.BuildRequest.get_by_instance_uuid(
            context, instance.uuid)
        # TODO(alaski): Sync API updates of the build_request to the
        # instance before it is destroyed. Right now only locked_by can
        # be updated before this is destroyed.
        build_request.destroy()

    def _populate_instance_mapping(self, context, instance, host):
        try:
            inst_mapping = objects.InstanceMapping.get_by_instance_uuid(
                    context, instance.uuid)
        except exception.InstanceMappingNotFound:
            # NOTE(alaski): If nova-api is up to date this exception should
            # never be hit. But during an upgrade it's possible that an old
            # nova-api didn't create an instance_mapping during this boot
            # request.
            LOG.debug('Instance was not mapped to a cell, likely due '
                      'to an older nova-api service running.',
                      instance=instance)
            return None
        else:
            try:
                host_mapping = objects.HostMapping.get_by_host(context,
                        host['host'])
            except exception.HostMappingNotFound:
                # NOTE(alaski): For now this exception means that a
                # deployment has not migrated to cellsv2 and we should
                # remove the instance_mapping that has been created.
                # Eventually this will indicate a failure to properly map a
                # host to a cell and we may want to reschedule.
                inst_mapping.destroy()
                return None
            else:
                inst_mapping.cell_mapping = host_mapping.cell_mapping
                inst_mapping.save()
        return inst_mapping

    # NOTE(danms): This is never cell-targeted because it is only used for
    # cellsv1 (which does not target cells directly) and n-cpu reschedules
    # (which go to the cell conductor and thus are always cell-specific).
    def build_instances(self, context, instances, image, filter_properties,
            admin_password, injected_files, requested_networks,
            security_groups, block_device_mapping=None, legacy_bdm=True):
        # TODO(ndipanov): Remove block_device_mapping and legacy_bdm in version
        #                 2.0 of the RPC API.
        # TODO(danms): Remove this in version 2.0 of the RPC API
        if (requested_networks and
                not isinstance(requested_networks,
                               objects.NetworkRequestList)):
            requested_networks = objects.NetworkRequestList.from_tuples(
                requested_networks)
        # TODO(melwitt): Remove this in version 2.0 of the RPC API
        flavor = filter_properties.get('instance_type')
        if flavor and not isinstance(flavor, objects.Flavor):
            # Code downstream may expect extra_specs to be populated since it
            # is receiving an object, so lookup the flavor to ensure this.
            flavor = objects.Flavor.get_by_id(context, flavor['id'])
            filter_properties = dict(filter_properties, instance_type=flavor)

        request_spec = {}
        is_bfv = (block_device_mapping.root_bdm_is_volume()
                  if block_device_mapping else False)
        try:
            # check retry policy. Rather ugly use of instances[0]...
            # but if we've exceeded max retries... then we really only
            # have a single instance.
            # TODO(sbauza): Provide directly the RequestSpec object
            # when _set_vm_state_and_notify() and populate_retry()
            # accept it
            request_spec = scheduler_utils.build_request_spec(
                context, image, instances)
            scheduler_utils.populate_retry(
                filter_properties, instances[0].uuid)
            instance_uuids = [instance.uuid for instance in instances]
            spec_obj = objects.RequestSpec.from_primitives(
                    context, request_spec, filter_properties)
            hosts = self._schedule_instances(
                    context, spec_obj, is_bfv, instance_uuids)
        except Exception as exc:
            num_attempts = filter_properties.get(
                'retry', {}).get('num_attempts', 1)
            updates = {'vm_state': vm_states.ERROR, 'task_state': None}
            for instance in instances:
                self._set_vm_state_and_notify(
                    context, instance.uuid, 'build_instances', updates,
                    exc, request_spec)
                # If num_attempts > 1, we're in a reschedule and probably
                # either hit NoValidHost or MaxRetriesExceeded. Either way,
                # the build request should already be gone and we probably
                # can't reach the API DB from the cell conductor.
                if num_attempts <= 1:
                    try:
                        # If the BuildRequest stays around then instance
                        # show/lists will pull from it rather than the errored
                        # instance.
                        self._destroy_build_request(context, instance)
                    except exception.BuildRequestNotFound:
                        pass
                self._cleanup_allocated_networks(
                    context, instance, requested_networks)
            return

        for (instance, host) in six.moves.zip(instances, hosts):
            instance.availability_zone = (
                availability_zones.get_host_availability_zone(context,
                                                              host['host']))
            try:
                # NOTE(danms): This saves the az change above, refreshes our
                # instance, and tells us if it has been deleted underneath us
                instance.save()
            except (exception.InstanceNotFound,
                    exception.InstanceInfoCacheNotFound):
                LOG.debug('Instance deleted during build', instance=instance)
                continue
            local_filter_props = copy.deepcopy(filter_properties)
            scheduler_utils.populate_filter_properties(local_filter_props,
                host)
            # The block_device_mapping passed from the api doesn't contain
            # instance specific information
            bdms = objects.BlockDeviceMappingList.get_by_instance_uuid(
                    context, instance.uuid)

            # This is populated in scheduler_utils.populate_retry
            num_attempts = local_filter_props.get('retry',
                                                  {}).get('num_attempts', 1)
            if num_attempts <= 1:
                # If this is a reschedule the instance is already mapped to
                # this cell and the BuildRequest is already deleted so ignore
                # the logic below.
                inst_mapping = self._populate_instance_mapping(context,
                                                               instance,
                                                               host)
                try:
                    self._destroy_build_request(context, instance)
                except exception.BuildRequestNotFound:
                    # This indicates an instance delete has been requested in
                    # the API. Stop the build, cleanup the instance_mapping and
                    # potentially the block_device_mappings
                    # TODO(alaski): Handle block_device_mapping cleanup
                    if inst_mapping:
                        inst_mapping.destroy()
                    return

            self.compute_rpcapi.build_and_run_instance(context,
                    instance=instance, host=host['host'], image=image,
                    request_spec=request_spec,
                    filter_properties=local_filter_props,
                    admin_password=admin_password,
                    injected_files=injected_files,
                    requested_networks=requested_networks,
                    security_groups=security_groups,
                    block_device_mapping=bdms, node=host['nodename'],
                    limits=host['limits'])

    def _schedule_instances(self, context, request_spec, is_bfv,
                            instance_uuids=None):
        scheduler_utils.setup_instance_group(context, request_spec)
        # NOTE(danms): We don't pass enough information to the scheduler to
        # know that we have a boot-from-volume request.
        # TODO(danms): We need to pass more context to the scheduler here
        # in order to (a) handle boot-from-volume instances, as well as
        # (b) know which volume provider to request resource from.
        request_spec_copy = request_spec
        if is_bfv:
            LOG.debug('Requesting zero root disk for '
                      'boot-from-volume instance')
            # Clone this so we don't mutate the RequestSpec that was passed in
            request_spec_copy = request_spec.obj_clone()
            request_spec_copy.flavor.root_gb = 0
        hosts = self.scheduler_client.select_destinations(context,
            request_spec_copy, instance_uuids)
        return hosts

    @targets_cell
    def unshelve_instance(self, context, instance, request_spec=None):
        sys_meta = instance.system_metadata

        def safe_image_show(ctx, image_id):
            if image_id:
                return self.image_api.get(ctx, image_id, show_deleted=False)
            else:
                raise exception.ImageNotFound(image_id='')

        if instance.vm_state == vm_states.SHELVED:
            instance.task_state = task_states.POWERING_ON
            instance.save(expected_task_state=task_states.UNSHELVING)
            self.compute_rpcapi.start_instance(context, instance)
        elif instance.vm_state == vm_states.SHELVED_OFFLOADED:
            image = None
            image_id = sys_meta.get('shelved_image_id')
            # No need to check for image if image_id is None as
            # "shelved_image_id" key is not set for volume backed
            # instance during the shelve process
            if image_id:
                with compute_utils.EventReporter(
                    context, 'get_image_info', instance.uuid):
                    try:
                        image = safe_image_show(context, image_id)
                    except exception.ImageNotFound:
                        instance.vm_state = vm_states.ERROR
                        instance.save()

                        reason = _('Unshelve attempted but the image %s '
                                   'cannot be found.') % image_id

                        LOG.error(reason, instance=instance)
                        raise exception.UnshelveException(
                            instance_id=instance.uuid, reason=reason)

            try:
                with compute_utils.EventReporter(context, 'schedule_instances',
                                                 instance.uuid):
                    if not request_spec:
                        # NOTE(sbauza): We were unable to find an original
                        # RequestSpec object - probably because the instance is
                        # old. We need to mock that the old way
                        filter_properties = {}
                        request_spec = scheduler_utils.build_request_spec(
                            context, image, [instance])
                    else:
                        # NOTE(sbauza): Force_hosts/nodes needs to be reset
                        # if we want to make sure that the next destination
                        # is not forced to be the original host
                        request_spec.reset_forced_destinations()
                        # TODO(sbauza): Provide directly the RequestSpec object
                        # when populate_filter_properties and populate_retry()
                        # accept it
                        filter_properties = request_spec.\
                            to_legacy_filter_properties_dict()
                        request_spec = request_spec.\
                            to_legacy_request_spec_dict()
                    scheduler_utils.populate_retry(filter_properties,
                                                   instance.uuid)
                    request_spec = objects.RequestSpec.from_primitives(
                        context, request_spec, filter_properties)
                    # NOTE(cfriesen): Ensure that we restrict the scheduler to
                    # the cell specified by the instance mapping.
                    instance_mapping = \
                        objects.InstanceMapping.get_by_instance_uuid(
                            context, instance.uuid)
                    LOG.debug('Requesting cell %(cell)s while unshelving',
                              {'cell': instance_mapping.cell_mapping.identity},
                              instance=instance)
                    if ('requested_destination' in request_spec and
                            request_spec.requested_destination):
                        request_spec.requested_destination.cell = (
                            instance_mapping.cell_mapping)
                    else:
                        request_spec.requested_destination = (
                            objects.Destination(
                                cell=instance_mapping.cell_mapping))

                    request_spec.ensure_project_id(instance)
                    hosts = self._schedule_instances(
                        context, request_spec, instance.is_volume_backed(),
                        [instance.uuid])
                    host_state = hosts[0]
                    scheduler_utils.populate_filter_properties(
                            filter_properties, host_state)
                    (host, node) = (host_state['host'], host_state['nodename'])
                    instance.availability_zone = (
                        availability_zones.get_host_availability_zone(
                            context, host))
                    self.compute_rpcapi.unshelve_instance(
                            context, instance, host, image=image,
                            filter_properties=filter_properties, node=node)
            except (exception.NoValidHost,
                    exception.UnsupportedPolicyException):
                instance.task_state = None
                instance.save()
                LOG.warning("No valid host found for unshelve instance",
                            instance=instance)
                return
            except Exception:
                with excutils.save_and_reraise_exception():
                    instance.task_state = None
                    instance.save()
                    LOG.error("Unshelve attempted but an error "
                              "has occurred", instance=instance)
        else:
            LOG.error('Unshelve attempted but vm_state not SHELVED or '
                      'SHELVED_OFFLOADED', instance=instance)
            instance.vm_state = vm_states.ERROR
            instance.save()
            return

    def _allocate_for_evacuate_dest_host(self, context, instance, host,
                                         request_spec=None):
        # The user is forcing the destination host and bypassing the
        # scheduler. We need to copy the source compute node
        # allocations in Placement to the destination compute node.
        # Normally select_destinations() in the scheduler would do this
        # for us, but when forcing the target host we don't call the
        # scheduler.
        source_node = None  # This is used for error handling below.
        try:
            source_node = objects.ComputeNode.get_by_host_and_nodename(
                context, instance.host, instance.node)
            dest_node = (
                objects.ComputeNode.get_first_node_by_host_for_old_compat(
                    context, host, use_slave=True))
        except exception.ComputeHostNotFound as ex:
            with excutils.save_and_reraise_exception():
                # TODO(mriedem): This ugly RequestSpec handling should be
                # tucked away in _set_vm_state_and_notify.
                if request_spec:
                    request_spec = \
                        request_spec.to_legacy_request_spec_dict()
                else:
                    request_spec = {}
                self._set_vm_state_and_notify(
                    context, instance.uuid, 'rebuild_server',
                    {'vm_state': instance.vm_state,
                     'task_state': None}, ex, request_spec)
                if source_node:
                    LOG.warning('Specified host %s for evacuate was not '
                                'found.', host, instance=instance)
                else:
                    LOG.warning('Source host %s and node %s for evacuate was '
                                'not found.', instance.host, instance.node,
                                instance=instance)

        # TODO(mriedem): In Queens, call select_destinations() with a
        # skip_filters=True flag so the scheduler does the work of
        # claiming resources on the destination in Placement but still
        # bypass the scheduler filters, which honors the 'force' flag
        # in the API.
        try:
            scheduler_utils.claim_resources_on_destination(
                self.scheduler_client.reportclient, instance,
                source_node, dest_node)
        except exception.NoValidHost as ex:
            with excutils.save_and_reraise_exception():
                # TODO(mriedem): This ugly RequestSpec handling should be
                # tucked away in _set_vm_state_and_notify.
                if request_spec:
                    request_spec = \
                        request_spec.to_legacy_request_spec_dict()
                else:
                    request_spec = {}
                self._set_vm_state_and_notify(
                    context, instance.uuid, 'rebuild_server',
                    {'vm_state': instance.vm_state,
                     'task_state': None}, ex, request_spec)
                LOG.warning('Specified host %s for evacuate is '
                            'invalid.', host, instance=instance)

    @targets_cell
    def rebuild_instance(self, context, instance, orig_image_ref, image_ref,
                         injected_files, new_pass, orig_sys_metadata,
                         bdms, recreate, on_shared_storage,
                         preserve_ephemeral=False, host=None,
                         request_spec=None):

        with compute_utils.EventReporter(context, 'rebuild_server',
                                          instance.uuid):
            node = limits = None

            try:
                migration = objects.Migration.get_by_instance_and_status(
                    context, instance.uuid, 'accepted')
            except exception.MigrationNotFoundByStatus:
                LOG.debug("No migration record for the rebuild/evacuate "
                          "request.", instance=instance)
                migration = None

            # The host variable is passed in two cases:
            # 1. rebuild - the instance.host is passed to rebuild on the
            #       same host and bypass the scheduler *unless* a new image
            #       was specified
            # 2. evacuate with specified host and force=True - the specified
            #       host is passed and is meant to bypass the scheduler.
            # NOTE(mriedem): This could be a lot more straight-forward if we
            # had separate methods for rebuild and evacuate...
            if host:
                # We only create a new allocation on the specified host if
                # we're doing an evacuate since that is a move operation.
                if host != instance.host:
                    # If a destination host is forced for evacuate, create
                    # allocations against it in Placement.
                    self._allocate_for_evacuate_dest_host(
                        context, instance, host, request_spec)
            else:
                # At this point, the user is either:
                #
                # 1. Doing a rebuild on the same host (not evacuate) and
                #    specified a new image.
                # 2. Evacuating and specified a host but are not forcing it.
                #
                # In either case, the API passes host=None but sets up the
                # RequestSpec.requested_destination field for the specified
                # host.
                if not request_spec:
                    # NOTE(sbauza): We were unable to find an original
                    # RequestSpec object - probably because the instance is old
                    # We need to mock that the old way
                    # TODO(sbauza): Provide directly the RequestSpec object
                    # when _set_vm_state_and_notify() accepts it
                    filter_properties = {'ignore_hosts': [instance.host]}
                    # build_request_spec expects a primitive image dict
                    image_meta = nova_object.obj_to_primitive(
                        instance.image_meta)
                    request_spec = scheduler_utils.build_request_spec(
                            context, image_meta, [instance])
                    request_spec = objects.RequestSpec.from_primitives(
                        context, request_spec, filter_properties)
                elif recreate:
                    # NOTE(sbauza): Augment the RequestSpec object by excluding
                    # the source host for avoiding the scheduler to pick it
                    # WRS: overwrite ignore_hosts list in RequestSpec with
                    # source host.  This drops any ignore_hosts from previous
                    # requests which could prevent evacuate from scheduling
                    # properly.
                    request_spec.ignore_hosts = [instance.host]
                    # WRS: The request_spec has stale flavor and numa_topology,
                    # so these fields must be updated. This occurs when we do
                    # an evacuation after a reverted resize.
                    request_spec.flavor = instance.flavor
                    request_spec.numa_topology = instance.numa_topology
                    # NOTE(sbauza): Force_hosts/nodes needs to be reset
                    # if we want to make sure that the next destination
                    # is not forced to be the original host
                    request_spec.reset_forced_destinations()
                    # WRS: determine offline cpus due to scaling to be used
                    # to calculate placement service resource claim in
                    # scheduler.
                    request_spec.offline_cpus = \
                        scheduler_utils.determine_offline_cpus(
                                  instance.flavor, instance.numa_topology)

                    # WRS: these hints are needed by the vcpu filter
                    hints = dict()
                    hints['task_state'] = [instance.task_state or ""]
                    hints['host'] = [instance.host or ""]
                    hints['node'] = [instance.node or ""]
                    if request_spec.obj_attr_is_set('scheduler_hints') and \
                            request_spec.scheduler_hints:
                        request_spec.scheduler_hints.update(hints)
                    else:
                        request_spec.scheduler_hints = hints

                try:
                    request_spec.ensure_project_id(instance)
                    hosts = self._schedule_instances(
                        context, request_spec, instance.is_volume_backed(),
                        [instance.uuid])
                    host_dict = hosts.pop(0)
                    host, node, limits = (host_dict['host'],
                                          host_dict['nodename'],
                                          host_dict['limits'])
                except exception.NoValidHost as ex:
                    if migration:
                        migration.status = 'error'
                        migration.save()
                    request_spec = request_spec.to_legacy_request_spec_dict()
                    with excutils.save_and_reraise_exception():
                        self._set_vm_state_and_notify(context, instance.uuid,
                                'rebuild_server',
                                {'vm_state': instance.vm_state,
                                 'task_state': None}, ex, request_spec)
                        LOG.warning("No valid host found for rebuild",
                                    instance=instance)
                except exception.UnsupportedPolicyException as ex:
                    if migration:
                        migration.status = 'error'
                        migration.save()
                    request_spec = request_spec.to_legacy_request_spec_dict()
                    with excutils.save_and_reraise_exception():
                        self._set_vm_state_and_notify(context, instance.uuid,
                                'rebuild_server',
                                {'vm_state': instance.vm_state,
                                 'task_state': None}, ex, request_spec)
                        LOG.warning("Server with unsupported policy "
                                    "cannot be rebuilt", instance=instance)

            compute_utils.notify_about_instance_usage(
                self.notifier, context, instance, "rebuild.scheduled")

            instance.availability_zone = (
                availability_zones.get_host_availability_zone(
                    context, host))

            self.compute_rpcapi.rebuild_instance(context,
                    instance=instance,
                    new_pass=new_pass,
                    injected_files=injected_files,
                    image_ref=image_ref,
                    orig_image_ref=orig_image_ref,
                    orig_sys_metadata=orig_sys_metadata,
                    bdms=bdms,
                    recreate=recreate,
                    on_shared_storage=on_shared_storage,
                    preserve_ephemeral=preserve_ephemeral,
                    migration=migration,
                    host=host, node=node, limits=limits)

    # TODO(avolkov): move method to bdm
    @staticmethod
    def _volume_size(instance_type, bdm):
        size = bdm.get('volume_size')
        # NOTE (ndipanov): inherit flavor size only for swap and ephemeral
        if (size is None and bdm.get('source_type') == 'blank' and
                bdm.get('destination_type') == 'local'):
            if bdm.get('guest_format') == 'swap':
                size = instance_type.get('swap', 0)
            else:
                size = instance_type.get('ephemeral_gb', 0)
        return size

    def _create_block_device_mapping(self, cell, instance_type, instance_uuid,
                                     block_device_mapping):
        """Create the BlockDeviceMapping objects in the db.

        This method makes a copy of the list in order to avoid using the same
        id field in case this is called for multiple instances.
        """
        LOG.debug("block_device_mapping %s", list(block_device_mapping),
                  instance_uuid=instance_uuid)
        instance_block_device_mapping = copy.deepcopy(block_device_mapping)
        for bdm in instance_block_device_mapping:
            bdm.volume_size = self._volume_size(instance_type, bdm)
            bdm.instance_uuid = instance_uuid
            with obj_target_cell(bdm, cell):
                bdm.update_or_create()
        return instance_block_device_mapping

    def _create_tags(self, context, instance_uuid, tags):
        """Create the Tags objects in the db."""
        if tags:
            tag_list = [tag.tag for tag in tags]
            instance_tags = objects.TagList.create(
                context, instance_uuid, tag_list)
            return instance_tags
        else:
            return tags

    def _bury_in_cell0(self, context, request_spec, exc,
                       build_requests=None, instances=None):
        """Ensure all provided build_requests and instances end up in cell0.

        Cell0 is the fake cell we schedule dead instances to when we can't
        schedule them somewhere real. Requests that don't yet have instances
        will get a new instance, created in cell0. Instances that have not yet
        been created will be created in cell0. All build requests are destroyed
        after we're done. Failure to delete a build request will trigger the
        instance deletion, just like the happy path in
        schedule_and_build_instances() below.
        """
        try:
            cell0 = objects.CellMapping.get_by_uuid(
                context, objects.CellMapping.CELL0_UUID)
        except exception.CellMappingNotFound:
            # Not yet setup for cellsv2. Instances will need to be written
            # to the configured database. This will become a deployment
            # error in Ocata.
            LOG.error('No cell mapping found for cell0 while '
                      'trying to record scheduling failure. '
                      'Setup is incomplete.')
            return

        build_requests = build_requests or []
        instances = instances or []
        instances_by_uuid = {inst.uuid: inst for inst in instances}
        for build_request in build_requests:
            # WRS: unreserve volumes now since volume info will be lost
            # once build_request is destroyed
            for bdm in build_request.block_device_mappings:
                if bdm.volume_id:
                    self.volume_api.unreserve_volume(context, bdm.volume_id)
            if build_request.instance_uuid not in instances_by_uuid:
                # This is an instance object with no matching db entry.
                instance = build_request.get_new_instance(context)
                instances_by_uuid[instance.uuid] = instance

        updates = {'vm_state': vm_states.ERROR, 'task_state': None}
        legacy_spec = request_spec.to_legacy_request_spec_dict()
        for instance in instances_by_uuid.values():
            with obj_target_cell(instance, cell0) as cctxt:
                instance.create()
                # Use the context targeted to cell0 here since the instance is
                # now in cell0.
                self._set_vm_state_and_notify(
                    cctxt, instance.uuid, 'build_instances', updates,
                    exc, legacy_spec)
                try:
                    # We don't need the cell0-targeted context here because the
                    # instance mapping is in the API DB.
                    inst_mapping = \
                        objects.InstanceMapping.get_by_instance_uuid(
                            context, instance.uuid)
                    inst_mapping.cell_mapping = cell0
                    inst_mapping.save()
                except exception.InstanceMappingNotFound:
                    pass

        for build_request in build_requests:
            try:
                build_request.destroy()
            except exception.BuildRequestNotFound:
                # Instance was deleted before we finished scheduling
                inst = instances_by_uuid[build_request.instance_uuid]
                with obj_target_cell(inst, cell0):
                    inst.destroy()

    def schedule_and_build_instances(self, context, build_requests,
                                     request_specs, image,
                                     admin_password, injected_files,
                                     requested_networks, block_device_mapping,
                                     tags=None):
        is_bfv = (block_device_mapping.root_bdm_is_volume()
                  if block_device_mapping else False)
        # Add all the UUIDs for the instances
        instance_uuids = [spec.instance_uuid for spec in request_specs]
        try:
            hosts = self._schedule_instances(context, request_specs[0], is_bfv,
                                             instance_uuids)
        except Exception as exc:
            LOG.exception('Failed to schedule instances')
            self._bury_in_cell0(context, request_specs[0], exc,
                                build_requests=build_requests)
            return

        host_mapping_cache = {}
        cell_mapping_cache = {}
        instances = []

        # WRS: get number of instances allowed to build, may need to adjust
        #      for maximum server group size
        instance_group = request_specs[0].instance_group
        metadetails = {}
        if instance_group:
            metadetails = instance_group.get('metadetails', {})
        group_exceed = int(metadetails.get('wrs-sg:group_exceed', 0))
        num_allowed = len(build_requests)
        num_allowed -= max(group_exceed, 0)
        num_destroyed = 0

        # WRS: need to keep track of remaining build requests to delete them.
        # Not using deep copy since source object does not allow member delete
        remaining_build_requests = []
        for build_request in build_requests:
            remaining_build_requests.append(build_request)

        for (build_request, request_spec, host) in six.moves.zip(
                build_requests, request_specs, hosts):
            del remaining_build_requests[0]
            instance = build_request.get_new_instance(context)
            # Convert host from the scheduler into a cell record
            if host['host'] not in host_mapping_cache:
                try:
                    host_mapping = objects.HostMapping.get_by_host(
                        context, host['host'])
                    host_mapping_cache[host['host']] = host_mapping
                except exception.HostMappingNotFound as exc:
                    LOG.error('No host-to-cell mapping found for selected '
                              'host %(host)s. Setup is incomplete.',
                              {'host': host['host']})
                    self._bury_in_cell0(context, request_spec, exc,
                                        build_requests=[build_request],
                                        instances=[instance])
                    # This is a placeholder in case the quota recheck fails.
                    instances.append(None)
                    continue
            else:
                host_mapping = host_mapping_cache[host['host']]

            cell = host_mapping.cell_mapping

            # Before we create the instance, let's make one final check that
            # the build request is still around and wasn't deleted by the user
            # already.
            try:
                objects.BuildRequest.get_by_instance_uuid(
                    context, instance.uuid)
            except exception.BuildRequestNotFound:
                # the build request is gone so we're done for this instance
                LOG.debug('While scheduling instance, the build request '
                          'was already deleted.', instance=instance)
                # This is a placeholder in case the quota recheck fails.
                instances.append(None)
                rc = self.scheduler_client.reportclient
                rc.delete_allocation_for_instance(instance.uuid)
                continue
            else:
                # WRS:extension - destroy instances execeeding number of
                # allowed instances
                if num_allowed == 0:
                    try:
                        build_request.destroy()
                    except exception.BuildRequestNotFound:
                        pass
                    num_destroyed += 1

                    try:
                        instance_mapping = objects.InstanceMapping\
                                  .get_by_instance_uuid(context, instance.uuid)
                    except exception.InstanceMappingNotFound:
                        pass
                    else:
                        instance_mapping.destroy()

                    instances.append(None)
                    rc = self.scheduler_client.reportclient
                    rc.delete_allocation_for_instance(instance.uuid)
                    continue
                num_allowed -= 1
                instance.availability_zone = (
                    availability_zones.get_host_availability_zone(
                        context, host['host']))
                with obj_target_cell(instance, cell):
                    instance.create()
                    instances.append(instance)
                    cell_mapping_cache[instance.uuid] = cell

        for build_request in remaining_build_requests:
            build_request.destroy()

        # NOTE(melwitt): We recheck the quota after creating the
        # objects to prevent users from allocating more resources
        # than their allowed quota in the event of a race. This is
        # configurable because it can be expensive if strict quota
        # limits are not required in a deployment.
        if CONF.quota.recheck_quota:
            try:
                compute_utils.check_num_instances_quota(
                    context, instance.flavor, 0, 0,
                    orig_num_req=len(build_requests))
            except exception.TooManyInstances as exc:
                with excutils.save_and_reraise_exception():
                    self._cleanup_build_artifacts(context, exc, instances,
                                                  build_requests,
                                                  request_specs,
                                                  cell_mapping_cache)

        for (build_request, request_spec, host, instance) in six.moves.zip(
                build_requests, request_specs, hosts, instances):
            if instance is None:
                # Skip placeholders that were buried in cell0 or had their
                # build requests deleted by the user before instance create.
                continue
            cell = cell_mapping_cache[instance.uuid]
            filter_props = request_spec.to_legacy_filter_properties_dict()
            scheduler_utils.populate_retry(filter_props, instance.uuid)
            scheduler_utils.populate_filter_properties(filter_props,
                                                       host)
            # TODO(melwitt): Maybe we should set_target_cell on the contexts
            # once we map to a cell, and remove these separate with statements.
            with obj_target_cell(instance, cell) as cctxt:
                # send a state update notification for the initial create to
                # show it going from non-existent to BUILDING
                # This can lazy-load attributes on instance.
                notifications.send_update_with_states(cctxt, instance, None,
                        vm_states.BUILDING, None, None, service="conductor")
                objects.InstanceAction.action_start(
                    cctxt, instance.uuid, instance_actions.CREATE,
                    want_result=False)
                instance_bdms = self._create_block_device_mapping(
                    cell, instance.flavor, instance.uuid, block_device_mapping)
                instance_tags = self._create_tags(cctxt, instance.uuid, tags)

            # TODO(Kevin Zheng): clean this up once instance.create() handles
            # tags; we do this so the instance.create notification in
            # build_and_run_instance in nova-compute doesn't lazy-load tags
            instance.tags = instance_tags if instance_tags \
                else objects.TagList()

            # Update mapping for instance. Normally this check is guarded by
            # a try/except but if we're here we know that a newer nova-api
            # handled the build process and would have created the mapping
            inst_mapping = objects.InstanceMapping.get_by_instance_uuid(
                context, instance.uuid)
            inst_mapping.cell_mapping = cell
            inst_mapping.save()

            if not self._delete_build_request(
                    context, build_request, instance, cell, instance_bdms,
                    instance_tags):
                # The build request was deleted before/during scheduling so
                # the instance is gone and we don't have anything to build for
                # this one.
                continue

            # NOTE(danms): Compute RPC expects security group names or ids
            # not objects, so convert this to a list of names until we can
            # pass the objects.
            legacy_secgroups = [s.identifier
                                for s in request_spec.security_groups]

            with obj_target_cell(instance, cell) as cctxt:
                self.compute_rpcapi.build_and_run_instance(
                    cctxt, instance=instance, image=image,
                    request_spec=request_spec,
                    filter_properties=filter_props,
                    admin_password=admin_password,
                    injected_files=injected_files,
                    requested_networks=requested_networks,
                    security_groups=legacy_secgroups,
                    block_device_mapping=instance_bdms,
                    host=host['host'], node=host['nodename'],
                    limits=host['limits'])

        # WRS:extension - rollback quotas for destroyed instances
        if num_destroyed > 0:
            # get data to rollback
            # this code is copied from _check_num_instances_quota()
            #       in compute/api.py
            flavor = request_specs[0].flavor
            unused_cores = num_destroyed * flavor['vcpus']
            unused_vram = int(flavor.get('extra_specs', {})
                              .get('hw_video:ram_max_mb', 0))
            unused_ram = num_destroyed * (flavor['memory_mb'] + unused_vram)
            # rollback quotas
            quotas = objects.Quotas(context)
            quotas.reserve(instances=-num_destroyed,
                           cores=-unused_cores,
                           ram=-unused_ram)
            quotas.commit()

    def _cleanup_build_artifacts(self, context, exc, instances, build_requests,
                                 request_specs, cell_mapping_cache):
        for (instance, build_request, request_spec) in six.moves.zip(
                instances, build_requests, request_specs):
            # Skip placeholders that were buried in cell0 or had their
            # build requests deleted by the user before instance create.
            if instance is None:
                continue
            updates = {'vm_state': vm_states.ERROR, 'task_state': None}
            legacy_spec = request_spec.to_legacy_request_spec_dict()
            cell = cell_mapping_cache[instance.uuid]
            with try_target_cell(context, cell) as cctxt:
                self._set_vm_state_and_notify(cctxt, instance.uuid,
                                              'build_instances', updates, exc,
                                              legacy_spec)

            # TODO(mnaser): The cell mapping should already be populated by
            #               this point to avoid setting it below here.
            inst_mapping = objects.InstanceMapping.get_by_instance_uuid(
                context, instance.uuid)
            inst_mapping.cell_mapping = cell
            inst_mapping.save()

            # Be paranoid about artifacts being deleted underneath us.
            try:
                build_request.destroy()
            except exception.BuildRequestNotFound:
                pass
            try:
                request_spec.destroy()
            except exception.RequestSpecNotFound:
                pass

    def _delete_build_request(self, context, build_request, instance, cell,
                              instance_bdms, instance_tags):
        """Delete a build request after creating the instance in the cell.

        This method handles cleaning up the instance in case the build request
        is already deleted by the time we try to delete it.

        :param context: the context of the request being handled
        :type context: nova.context.RequestContext
        :param build_request: the build request to delete
        :type build_request: nova.objects.BuildRequest
        :param instance: the instance created from the build_request
        :type instance: nova.objects.Instance
        :param cell: the cell in which the instance was created
        :type cell: nova.objects.CellMapping
        :param instance_bdms: list of block device mappings for the instance
        :type instance_bdms: nova.objects.BlockDeviceMappingList
        :param instance_tags: list of tags for the instance
        :type instance_tags: nova.objects.TagList
        :returns: True if the build request was successfully deleted, False if
            the build request was already deleted and the instance is now gone.
        """
        try:
            build_request.destroy()
        except exception.BuildRequestNotFound:
            # This indicates an instance deletion request has been
            # processed, and the build should halt here. Clean up the
            # bdm, tags and instance record.
            with obj_target_cell(instance, cell) as cctxt:
                with compute_utils.notify_about_instance_delete(
                        self.notifier, cctxt, instance):
                    try:
                        instance.destroy()
                    except exception.InstanceNotFound:
                        pass
                    except exception.ObjectActionError:
                        # NOTE(melwitt): Instance became scheduled during
                        # the destroy, "host changed". Refresh and re-destroy.
                        try:
                            instance.refresh()
                            instance.destroy()
                        except exception.InstanceNotFound:
                            pass
            for bdm in instance_bdms:
                with obj_target_cell(bdm, cell):
                    try:
                        bdm.destroy()
                    except exception.ObjectActionError:
                        pass
            if instance_tags:
                with try_target_cell(context, cell) as target_ctxt:
                    try:
                        objects.TagList.destroy(target_ctxt, instance.uuid)
                    except exception.InstanceNotFound:
                        pass
            return False
        return True
