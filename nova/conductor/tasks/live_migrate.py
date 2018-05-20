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
# Copyright (c) 2016-2017 Wind River Systems, Inc.
#

from oslo_concurrency import lockutils
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_utils import strutils
import six

from nova.compute import power_state
from nova.compute import utils as compute_utils
from nova.conductor.tasks import base
import nova.conf
from nova import exception
from nova.i18n import _
from nova import objects
from nova.scheduler import utils as scheduler_utils
from nova import utils

LOG = logging.getLogger(__name__)
CONF = nova.conf.CONF


class LiveMigrationTask(base.TaskBase):
    def __init__(self, context, instance, destination,
                 block_migration, disk_over_commit, migration, compute_rpcapi,
                 servicegroup_api, scheduler_client, request_spec=None):
        super(LiveMigrationTask, self).__init__(context, instance)
        self.destination = destination
        self.sched_limits = None
        self.block_migration = block_migration
        self.disk_over_commit = disk_over_commit
        self.migration = migration
        self.source = instance.host
        self.migrate_data = None

        self.compute_rpcapi = compute_rpcapi
        self.servicegroup_api = servicegroup_api
        self.scheduler_client = scheduler_client
        self.request_spec = request_spec

    def _execute(self):
        self._check_instance_is_active()
        self._check_host_is_up(self.source)

        def _select_destination():
            if not self.destination:
                # Either no host was specified in the API request and the user
                # wants the scheduler to pick a destination host, or a host was
                # specified but is not forcing it, so they want the scheduler
                # filters to run on the specified host, like a scheduler hint.
                self.destination, self.sched_limits = self._find_destination()
            else:
                # This is the case that the user specified the 'force' flag
                # when live migrating with a specific destination host so the
                # scheduler is bypassed. There are still some minimal checks
                # performed here though.
                source_node, dest_node = self._check_requested_destination()
                # Now that we're semi-confident in the force specified host, we
                # need to copy the source compute node allocations in Placement
                # to the destination compute node.
                # Normally select_destinations()
                # in the scheduler would do this for us, but when forcing the
                # target host we don't call the scheduler.
                # TODO(mriedem): In Queens, call select_destinations() with a
                # skip_filters=True flag so the scheduler does the work of
                # claiming resources on the destination in Placement but still
                # bypass the scheduler filters, which honors the 'force' flag
                # in the API.
                # This raises NoValidHost which will be handled in
                # ComputeTaskManager.
                scheduler_utils.claim_resources_on_destination(
                    self.scheduler_client.reportclient, self.instance,
                    source_node, dest_node)

        if self._is_ordered_scheduling_needed():
            # WRS: ensure scheduling of one live-migration at a time for
            # instances in a given anti-affinity server group.
            # This closes a race condition.
            instance_group_name = self.request_spec.instance_group['name']

            sema = lockutils.lock('instance-group-%s' % instance_group_name,
                         external=True, fair=True)
        else:
            sema = compute_utils.UnlimitedSemaphore()

        with sema:
            _select_destination()

        # WRS: Log live migration
        LOG.info("Live migrating instance %(inst_uuid)s: "
                 "source:%(source)s dest:%(dest)s",
                 {'inst_uuid': self.instance['uuid'],
                  'source': self.source,
                  'dest': self.destination})

        # TODO(johngarbutt) need to move complexity out of compute manager
        # TODO(johngarbutt) disk_over_commit?
        return self.compute_rpcapi.live_migration(self.context,
                host=self.source,
                instance=self.instance,
                dest=self.destination,
                block_migration=self.block_migration,
                migration=self.migration,
                migrate_data=self.migrate_data)

    def rollback(self):
        # TODO(johngarbutt) need to implement the clean up operation
        # but this will make sense only once we pull in the compute
        # calls, since this class currently makes no state changes,
        # except to call the compute method, that has no matching
        # rollback call right now.
        pass

    def _is_ordered_scheduling_needed(self):
        if hasattr(self.request_spec, 'instance_group') and \
                   self.request_spec.instance_group:
            metadetails = self.request_spec.instance_group['metadetails']
            is_best_effort = strutils.bool_from_string(
                                metadetails.get('wrs-sg:best_effort', 'False'))

            if ('anti-affinity' in
                    self.request_spec.instance_group['policies'] and
                    not is_best_effort):
                return True

    def _check_instance_is_active(self):
        if self.instance.power_state not in (power_state.RUNNING,
                                             power_state.PAUSED):
            raise exception.InstanceInvalidState(
                    instance_uuid=self.instance.uuid,
                    attr='power_state',
                    state=self.instance.power_state,
                    method='live migrate')

    def _check_host_is_up(self, host):
        service = objects.Service.get_by_compute_host(self.context, host)

        if not self.servicegroup_api.service_is_up(service):
            raise exception.ComputeServiceUnavailable(host=host)

    def _check_requested_destination(self):
        """Performs basic pre-live migration checks for the forced host.

        :returns: tuple of (source ComputeNode, destination ComputeNode)
        """
        self._check_destination_is_not_source()
        self._check_host_is_up(self.destination)
        self._check_destination_has_enough_memory()
        source_node, dest_node = self._check_compatible_with_source_hypervisor(
            self.destination)
        self._call_livem_checks_on_host(self.destination,
                                        limits=self.sched_limits)
        # Make sure the forced destination host is in the same cell that the
        # instance currently lives in.
        # NOTE(mriedem): This can go away if/when the forced destination host
        # case calls select_destinations.
        source_cell_mapping = self._get_source_cell_mapping()
        dest_cell_mapping = self._get_destination_cell_mapping()
        if source_cell_mapping.uuid != dest_cell_mapping.uuid:
            raise exception.MigrationPreCheckError(
                reason=(_('Unable to force live migrate instance %s '
                          'across cells.') % self.instance.uuid))
        return source_node, dest_node

    def _check_destination_is_not_source(self):
        if self.destination == self.source:
            raise exception.UnableToMigrateToSelf(
                    instance_id=self.instance.uuid, host=self.destination)

    def _check_destination_has_enough_memory(self):
        # TODO(mriedem): This method can be removed when the forced host
        # scenario is calling select_destinations() in the scheduler because
        # Placement will be used to filter allocation candidates by MEMORY_MB.
        # We likely can't remove it until the CachingScheduler is gone though
        # since the CachingScheduler does not use Placement.
        compute = self._get_compute_info(self.destination)
        free_ram_mb = compute.free_ram_mb
        total_ram_mb = compute.memory_mb
        mem_inst = self.instance.memory_mb
        # NOTE(sbauza): Now the ComputeNode object reports an allocation ratio
        # that can be provided by the compute_node if new or by the controller
        ram_ratio = compute.ram_allocation_ratio

        # NOTE(sbauza): Mimic the RAMFilter logic in order to have the same
        # ram validation
        avail = total_ram_mb * ram_ratio - (total_ram_mb - free_ram_mb)
        if not mem_inst or avail <= mem_inst:
            instance_uuid = self.instance.uuid
            dest = self.destination
            reason = _("Unable to migrate %(instance_uuid)s to %(dest)s: "
                       "Lack of memory(host:%(avail)s <= "
                       "instance:%(mem_inst)s)")
            raise exception.MigrationPreCheckError(reason=reason % dict(
                    instance_uuid=instance_uuid, dest=dest, avail=avail,
                    mem_inst=mem_inst))

    def _get_compute_info(self, host):
        return objects.ComputeNode.get_first_node_by_host_for_old_compat(
            self.context, host)

    def _check_compatible_with_source_hypervisor(self, destination):
        source_info = self._get_compute_info(self.source)
        destination_info = self._get_compute_info(destination)

        source_type = source_info.hypervisor_type
        destination_type = destination_info.hypervisor_type
        if source_type != destination_type:
            raise exception.InvalidHypervisorType()

        source_version = source_info.hypervisor_version
        destination_version = destination_info.hypervisor_version
        if source_version > destination_version:
            raise exception.DestinationHypervisorTooOld()
        return source_info, destination_info

    def _call_livem_checks_on_host(self, destination, limits=None):
        try:
            self.migrate_data = self.compute_rpcapi.\
                check_can_live_migrate_destination(self.context, self.instance,
                    destination, self.block_migration, self.disk_over_commit,
                    migration=self.migration, limits=limits)
        except messaging.MessagingTimeout:
            msg = _("Timeout while checking if we can live migrate to host: "
                    "%s") % destination
            raise exception.MigrationPreCheckError(msg)

    def _get_source_cell_mapping(self):
        """Returns the CellMapping for the cell in which the instance lives

        :returns: nova.objects.CellMapping record for the cell where
            the instance currently lives.
        :raises: MigrationPreCheckError - in case a mapping is not found
        """
        try:
            return objects.InstanceMapping.get_by_instance_uuid(
                self.context, self.instance.uuid).cell_mapping
        except exception.InstanceMappingNotFound:
            raise exception.MigrationPreCheckError(
                reason=(_('Unable to determine in which cell '
                          'instance %s lives.') % self.instance.uuid))

    def _get_destination_cell_mapping(self):
        """Returns the CellMapping for the destination host

        :returns: nova.objects.CellMapping record for the cell where
            the destination host is mapped.
        :raises: MigrationPreCheckError - in case a mapping is not found
        """
        try:
            return objects.HostMapping.get_by_host(
                self.context, self.destination).cell_mapping
        except exception.HostMappingNotFound:
            raise exception.MigrationPreCheckError(
                reason=(_('Unable to determine in which cell '
                          'destination host %s lives.') % self.destination))

    def _find_destination(self):
        # TODO(johngarbutt) this retry loop should be shared
        attempted_hosts = [self.source]
        image = utils.get_image_from_system_metadata(
            self.instance.system_metadata)
        filter_properties = {'ignore_hosts': attempted_hosts}
        if not self.request_spec:
            # NOTE(sbauza): We were unable to find an original RequestSpec
            # object - probably because the instance is old.
            # We need to mock that the old way

            # WRS: these hints are needed by the vcpu filter
            hints = filter_properties.get('scheduler_hints', {})
            hints['task_state'] = self.instance.task_state or ""
            hints['host'] = self.instance.host or ""
            hints['node'] = self.instance.node or ""
            filter_properties['scheduler_hints'] = hints

            request_spec = objects.RequestSpec.from_components(
                self.context, self.instance.uuid, image,
                self.instance.flavor, self.instance.numa_topology,
                self.instance.pci_requests,
                filter_properties, None, self.instance.availability_zone
            )
        else:
            request_spec = self.request_spec
            # WRS: these hints are needed by the vcpu filter
            hints = dict()
            hints['task_state'] = [self.instance.task_state or ""]
            hints['host'] = [self.instance.host or ""]
            hints['node'] = [self.instance.node or ""]
            if request_spec.obj_attr_is_set('scheduler_hints') and \
                request_spec.scheduler_hints:
                request_spec.scheduler_hints.update(hints)
            else:
                request_spec.scheduler_hints = hints

            # NOTE(sbauza): Force_hosts/nodes needs to be reset
            # if we want to make sure that the next destination
            # is not forced to be the original host
            request_spec.reset_forced_destinations()

            # WRS: The request_spec has stale flavor, so this field must be
            # updated. This occurs when we do a live-migration after a resize.
            request_spec.flavor = self.instance.flavor

            # WRS: The request_spec has stale instance_group information.
            # Update from db to get latest members and metadetails.
            if hasattr(request_spec, 'instance_group') and \
                       request_spec.instance_group:
                request_spec.instance_group = \
                    objects.InstanceGroup.get_by_instance_uuid(
                           self.context, self.instance.uuid)

                # WRS: add hosts to Server group host list for group members
                # that are migrating in progress
                metadetails = request_spec.instance_group['metadetails']
                is_best_effort = strutils.bool_from_string(
                    metadetails.get('wrs-sg:best_effort', 'False'))

                if ('anti-affinity' in request_spec.instance_group['policies']
                    and not is_best_effort):
                    group_members = request_spec.instance_group['members']

                    for member_uuid in group_members:
                        if member_uuid == self.instance.uuid:
                            continue
                        filters = {
                            'migration_type': 'live-migration',
                            'instance_uuid': member_uuid,
                            'status': ['queued', 'accepted', 'pre-migrating',
                                       'preparing', 'running']
                        }
                        migrations = objects.MigrationList. \
                            get_by_filters(self.context, filters)

                        for migration in migrations:
                            if migration['source_compute'] not in \
                                    request_spec.instance_group['hosts']:
                                request_spec.instance_group['hosts'].\
                                    append(migration['source_compute'])
                            if (migration['dest_compute'] and (
                                migration['dest_compute'] not in
                                        request_spec.instance_group['hosts'])):
                                request_spec.instance_group['hosts'].\
                                    append(migration['dest_compute'])

        scheduler_utils.setup_instance_group(self.context, request_spec)

        # We currently only support live migrating to hosts in the same
        # cell that the instance lives in, so we need to tell the scheduler
        # to limit the applicable hosts based on cell.
        cell_mapping = self._get_source_cell_mapping()
        LOG.debug('Requesting cell %(cell)s while live migrating',
                  {'cell': cell_mapping.identity},
                  instance=self.instance)
        if ('requested_destination' in request_spec and
                request_spec.requested_destination):
            request_spec.requested_destination.cell = cell_mapping
        else:
            request_spec.requested_destination = objects.Destination(
                cell=cell_mapping)

        request_spec.ensure_project_id(self.instance)

        # WRS: determine offline cpus due to scaling to be used to calculate
        # placement service resource claim in scheduler
        request_spec.offline_cpus = scheduler_utils.determine_offline_cpus(
                         self.instance.flavor, self.instance.numa_topology)
        host = limits = None
        migration_error = {}
        while host is None:
            self._check_not_over_max_retries(attempted_hosts)
            request_spec.ignore_hosts = attempted_hosts
            try:
                # WRS: determine if instance is volume backed and update
                # request spec to avoid allocating local disk resources.
                request_spec_copy = request_spec
                if self.instance.is_volume_backed():
                    LOG.debug('Requesting zero root disk for '
                              'boot-from-volume instance')
                    # Clone this so we don't mutate the RequestSpec that was
                    # passed in
                    request_spec_copy = request_spec.obj_clone()
                    request_spec_copy.flavor.root_gb = 0
                hoststate = self.scheduler_client.select_destinations(
                    self.context, request_spec_copy, [self.instance.uuid])[0]
                host = hoststate['host']
                limits = hoststate['limits']
            except messaging.RemoteError as ex:
                # TODO(ShaoHe Feng) There maybe multi-scheduler, and the
                # scheduling algorithm is R-R, we can let other scheduler try.
                # Note(ShaoHe Feng) There are types of RemoteError, such as
                # NoSuchMethod, UnsupportedVersion, we can distinguish it by
                # ex.exc_type.
                raise exception.MigrationSchedulerRPCError(
                    reason=six.text_type(ex))
            except exception.NoValidHost as ex:
                if (migration_error):
                    # remove duplicated and superfluous info from exception
                    msg = "%s" % ex.message
                    msg = msg.replace("No valid host was found.", "")
                    msg = msg.replace("No filter information", "")
                    fp = {'reject_map': migration_error}
                    scheduler_utils.NoValidHost_extend(fp, reason=msg)
                else:
                    raise
            try:
                self._check_compatible_with_source_hypervisor(host)
                # NOTE(ndipanov): We don't need to pass the node as it's not
                # relevant for drivers that support live migration
                self._call_livem_checks_on_host(host, limits=limits)
            except (exception.Invalid,
                    exception.MigrationPreCheckError) as e:
                # WRS: Change this from 'debug' log to 'info', we need this.
                LOG.info("Skipping host: %(host)s because: %(e)s",
                         {"host": host, "e": e})
                migration_error[host] = "%s" % e.message
                attempted_hosts.append(host)
                # The scheduler would have created allocations against the
                # selected destination host in Placement, so we need to remove
                # those before moving on.
                self._remove_host_allocations(host, hoststate['nodename'],
                                              request_spec)
                host = limits = None
        return host, limits

    def _remove_host_allocations(self, host, node, request_spec):
        """Removes instance allocations against the given host from Placement

        :param host: The name of the host.
        :param node: The name of the node.
        """
        # Get the compute node object since we need the UUID.
        # TODO(mriedem): If the result of select_destinations eventually
        # returns the compute node uuid, we wouldn't need to look it
        # up via host/node and we can save some time.
        try:
            compute_node = objects.ComputeNode.get_by_host_and_nodename(
                self.context, host, node)
        except exception.ComputeHostNotFound:
            # This shouldn't happen, but we're being careful.
            LOG.info('Unable to remove instance allocations from host %s '
                     'and node %s since it was not found.', host, node,
                     instance=self.instance)
            return

        # Calculate the resource class amounts to subtract from the allocations
        # on the node based on the instance flavor.
        resources = scheduler_utils.resources_from_flavor(
            self.instance, self.instance.flavor)

        # WRS: adjust resource allocations based on request_spec
        vcpus = request_spec.flavor.vcpus - request_spec.offline_cpus
        extra_specs = request_spec.flavor.extra_specs
        image_props = request_spec.image.properties
        instance_numa_topology = request_spec.numa_topology
        normalized_resources = \
                  scheduler_utils.normalized_resources_for_placement_claim(
                             resources, compute_node, vcpus, extra_specs,
                             image_props, instance_numa_topology)

        # Now remove the allocations for our instance against that node.
        # Note that this does not remove allocations against any other node
        # or shared resource provider, it's just undoing what the scheduler
        # allocated for the given (destination) node.
        self.scheduler_client.reportclient.\
            remove_provider_from_instance_allocation(
                self.instance.uuid, compute_node.uuid, self.instance.user_id,
                self.instance.project_id, normalized_resources)

    def _check_not_over_max_retries(self, attempted_hosts):
        if CONF.migrate_max_retries == -1:
            return

        retries = len(attempted_hosts) - 1
        if retries > CONF.migrate_max_retries:
            if self.migration:
                self.migration.status = 'failed'
                self.migration.save()
            msg = (_('Exceeded max scheduling retries %(max_retries)d for '
                     'instance %(instance_uuid)s during live migration')
                   % {'max_retries': retries,
                      'instance_uuid': self.instance.uuid})
            raise exception.MaxRetriesExceeded(reason=msg)
