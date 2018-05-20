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

from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import strutils

from nova import availability_zones
from nova.conductor.tasks import base
from nova import objects
from nova.scheduler import utils as scheduler_utils

LOG = logging.getLogger(__name__)


class MigrationTask(base.TaskBase):
    def __init__(self, context, instance, flavor,
                 request_spec, reservations, clean_shutdown, compute_rpcapi,
                 scheduler_client):
        super(MigrationTask, self).__init__(context, instance)
        self.clean_shutdown = clean_shutdown
        self.request_spec = request_spec
        self.reservations = reservations
        self.flavor = flavor

        self.compute_rpcapi = compute_rpcapi
        self.scheduler_client = scheduler_client

    def _execute(self):
        # TODO(sbauza): Remove that once prep_resize() accepts a  RequestSpec
        # object in the signature and all the scheduler.utils methods too
        legacy_spec = self.request_spec.to_legacy_request_spec_dict()
        legacy_props = self.request_spec.to_legacy_filter_properties_dict()
        scheduler_utils.setup_instance_group(self.context, self.request_spec)

        # WRS: add hosts to Server group host list for group members
        # that are migrating in progress
        if 'group_members' in legacy_props:
            metadetails = self.request_spec.instance_group['metadetails']
            is_best_effort = strutils.bool_from_string(
                metadetails.get('wrs-sg:best_effort', 'False'))

            if ('anti-affinity' in self.request_spec.instance_group['policies']
                    and not is_best_effort):
                group_members = self.request_spec.instance_group['members']

                for instance_uuid in group_members:
                    filters = {
                        'migration_type': 'migration',
                        'instance_uuid': instance_uuid,
                        'status': ['queued', 'pre-migrating', 'migrating',
                                   'post-migrating', 'finished']
                    }

                    migrations = objects.MigrationList.get_by_filters(
                                                         self.context, filters)

                    for migration in migrations:
                        if migration['source_compute'] not in \
                                self.request_spec.instance_group['hosts']:
                            self.request_spec.instance_group['hosts'].append(
                                                   migration['source_compute'])
                        if (migration['dest_compute'] and (
                                migration['dest_compute'] not in
                                   self.request_spec.instance_group['hosts'])):
                            self.request_spec.instance_group['hosts'].append(
                                                     migration['dest_compute'])

                # refresh legacy_spec and legacy_props with latest request_spec
                legacy_spec = self.request_spec.to_legacy_request_spec_dict()
                legacy_props = self.\
                    request_spec.to_legacy_filter_properties_dict()

        scheduler_utils.populate_retry(legacy_props,
                                       self.instance.uuid)

        # NOTE(sbauza): Force_hosts/nodes needs to be reset
        # if we want to make sure that the next destination
        # is not forced to be the original host
        self.request_spec.reset_forced_destinations()

        # NOTE(danms): Right now we only support migrate to the same
        # cell as the current instance, so request that the scheduler
        # limit thusly.
        instance_mapping = objects.InstanceMapping.get_by_instance_uuid(
            self.context, self.instance.uuid)
        LOG.debug('Requesting cell %(cell)s while migrating',
                  {'cell': instance_mapping.cell_mapping.identity},
                  instance=self.instance)
        if ('requested_destination' in self.request_spec and
                self.request_spec.requested_destination):
            self.request_spec.requested_destination.cell = (
                instance_mapping.cell_mapping)
        else:
            self.request_spec.requested_destination = objects.Destination(
                cell=instance_mapping.cell_mapping)

        self.request_spec.ensure_project_id(self.instance)

        # WRS: determine offline cpus due to scaling to be used to calculate
        # placement service resource claim in scheduler
        self.request_spec.offline_cpus = \
                  scheduler_utils.determine_offline_cpus(
                         self.flavor, self.instance.numa_topology)
        # NOTE(danms): We don't pass enough information to the scheduler to
        # know that we have a boot-from-volume request.
        # TODO(danms): We need to pass more context to the scheduler here
        # in order to (a) handle boot-from-volume instances, as well as
        # (b) know which volume provider to request resource from.
        request_spec_copy = self.request_spec
        if self.instance.is_volume_backed():
            LOG.debug('Requesting zero root disk for '
                      'boot-from-volume instance')
            # Clone this so we don't mutate the RequestSpec that was passed in
            request_spec_copy = self.request_spec.obj_clone()
            request_spec_copy.flavor.root_gb = 0

        hosts = self.scheduler_client.select_destinations(
            self.context, request_spec_copy, [self.instance.uuid])
        host_state = hosts[0]

        scheduler_utils.populate_filter_properties(legacy_props,
                                                   host_state)
        # context is not serializable
        legacy_props.pop('context', None)

        (host, node) = (host_state['host'], host_state['nodename'])

        self.instance.availability_zone = (
            availability_zones.get_host_availability_zone(
                self.context, host))

        # FIXME(sbauza): Serialize/Unserialize the legacy dict because of
        # oslo.messaging #1529084 to transform datetime values into strings.
        # tl;dr: datetimes in dicts are not accepted as correct values by the
        # rpc fake driver.
        legacy_spec = jsonutils.loads(jsonutils.dumps(legacy_spec))

        self.compute_rpcapi.prep_resize(
            self.context, self.instance, legacy_spec['image'],
            self.flavor, host, self.reservations,
            request_spec=legacy_spec, filter_properties=legacy_props,
            node=node, clean_shutdown=self.clean_shutdown)

        # WRS: return request_spec for save to db but need to clear retry and
        # instance_group hosts so that next request starts cleanly
        self.request_spec.retry = None
        if self.request_spec.instance_group:
            self.request_spec.instance_group.hosts = []
        return self.request_spec

    def rollback(self):
        pass
