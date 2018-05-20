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

import mock
from oslo_serialization import jsonutils

from nova import availability_zones
from nova.compute import rpcapi as compute_rpcapi
from nova.conductor.tasks import migrate
from nova import objects
from nova.scheduler import client as scheduler_client
from nova.scheduler import utils as scheduler_utils
from nova import test
from nova.tests.unit.conductor.test_conductor import FakeContext
from nova.tests.unit import fake_flavor
from nova.tests.unit import fake_instance


class MigrationTaskTestCase(test.NoDBTestCase):
    def setUp(self):
        super(MigrationTaskTestCase, self).setUp()
        self.user_id = 'fake'
        self.project_id = 'fake'
        self.context = FakeContext(self.user_id, self.project_id)
        self.flavor = fake_flavor.fake_flavor_obj(self.context)
        self.flavor.extra_specs = {'extra_specs': 'fake'}
        inst = fake_instance.fake_db_instance(image_ref='image_ref',
                                              instance_type=self.flavor)
        inst_object = objects.Instance(
            flavor=self.flavor,
            numa_topology=None,
            pci_requests=None,
            system_metadata={'image_hw_disk_bus': 'scsi'})
        self.instance = objects.Instance._from_db_object(
            self.context, inst_object, inst, [])
        self.request_spec = objects.RequestSpec(image=objects.ImageMeta(
                                    properties=objects.ImageMetaProps()),
                                                flavor=self.flavor)
        self.request_spec.instance_group = None
        self.hosts = [dict(host='host1', nodename=None, limits={})]
        self.filter_properties = {'limits': {}, 'retry': {'num_attempts': 1,
                                  'hosts': [['host1', None]]}}

        self.instance_group = objects.InstanceGroup()
        self.instance_group['metadetails'] = {'wrs-sg:best_effort': 'false',
                                              'wrs-sg:group_size': '2'}
        self.instance_group['members'] = ['uuid1', 'uuid2']
        self.instance_group['hosts'] = ['compute1', 'compute2']
        self.instance_group['policies'] = ['anti-affinity']

        self.reservations = []
        self.clean_shutdown = True

    def _stub_migrations(self, context, filters, dummy):
        fake_migrations = [
            {
                'id': 1234,
                'source_node': 'node1',
                'dest_node': 'node3',
                'source_compute': 'compute1',
                'dest_compute': 'compute3',
                'dest_host': '1.2.3.4',
                'status': 'post-migrating',
                'instance_uuid': 'uuid1',
                'old_instance_type_id': 1,
                'new_instance_type_id': 2,
                'migration_type': 'resize',
                'hidden': False,
                'created_at': '2017-05-29 13:42:02',
                'updated_at': '2017-05-29 13:42:02',
                'deleted_at': None,
                'deleted': False
            }
        ]
        return fake_migrations

    def _generate_task(self):
        return migrate.MigrationTask(self.context, self.instance, self.flavor,
                                     self.request_spec, self.reservations,
                                     self.clean_shutdown,
                                     compute_rpcapi.ComputeAPI(),
                                     scheduler_client.SchedulerClient())

    @mock.patch('nova.objects.RequestSpec.obj_clone')
    @mock.patch('nova.objects.Instance.is_volume_backed')
    @mock.patch('nova.availability_zones.get_host_availability_zone')
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    @mock.patch.object(scheduler_client.SchedulerClient, 'select_destinations')
    @mock.patch.object(compute_rpcapi.ComputeAPI, 'prep_resize')
    def test_execute(self, prep_resize_mock, sel_dest_mock, sig_mock, az_mock,
                     is_bfv_mock, clone_mock):
        sel_dest_mock.return_value = self.hosts
        az_mock.return_value = 'myaz'
        task = self._generate_task()
        legacy_request_spec = self.request_spec.to_legacy_request_spec_dict()
        # FIXME(sbauza): Serialize/Unserialize the legacy dict because of
        # oslo.messaging #1529084 to transform datetime values into strings.
        # tl;dr: datetimes in dicts are not accepted as correct values by the
        # rpc fake driver.
        legacy_request_spec_reloaded = jsonutils.loads(
            jsonutils.dumps(legacy_request_spec))
        task.execute()

        sig_mock.assert_called_once_with(self.context, self.request_spec)
        task.scheduler_client.select_destinations.assert_called_once_with(
            self.context, clone_mock.return_value, [self.instance.uuid])
        prep_resize_mock.assert_called_once_with(
            self.context, self.instance, legacy_request_spec['image'],
            self.flavor, self.hosts[0]['host'], self.reservations,
            request_spec=legacy_request_spec_reloaded,
            filter_properties=self.filter_properties,
            node=self.hosts[0]['nodename'], clean_shutdown=self.clean_shutdown)
        az_mock.assert_called_once_with(self.context, 'host1')

    @mock.patch('nova.objects.RequestSpec.obj_clone')
    @mock.patch('nova.objects.Instance.is_volume_backed')
    @mock.patch.object(availability_zones, 'get_host_availability_zone')
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    @mock.patch.object(scheduler_client.SchedulerClient, 'select_destinations')
    @mock.patch.object(compute_rpcapi.ComputeAPI, 'prep_resize')
    def test_execute_with_instance_group(self, prep_resize_mock,
                     sel_dest_mock, sig_mock, mock_get_avail_zone,
                     is_bfv_mock, clone_mock):
        sel_dest_mock.return_value = self.hosts
        mock_get_avail_zone.return_value = 'nova'

        filter_properties = {'instance_group': self.instance_group}
        self.request_spec._populate_group_info(filter_properties)

        # test for the addition of an extra Server group host:
        self.stub_out('nova.objects.migration.MigrationList.get_by_filters',
                      self._stub_migrations)

        task = self._generate_task()
        legacy_request_spec = self.request_spec.to_legacy_request_spec_dict()
        # FIXME(sbauza): Serialize/Unserialize the legacy dict because of
        # oslo.messaging #1529084 to transform datetime values into strings.
        # tl;dr: datetimes in dicts are not accepted as correct values by the
        # rpc fake driver.
        legacy_request_spec_reloaded = jsonutils.loads(
            jsonutils.dumps(legacy_request_spec))

        self.filter_properties = \
            {
                'limits': {},
                'retry': {'hosts': [['host1', None]], 'num_attempts': 1},
                'group_updated': True,
                'group_policies': {'anti-affinity'},
                'group_members': {'uuid1', 'uuid2'},
                'group_hosts': {'compute1', 'compute2', 'compute3'},
                'group_metadetails': {'wrs-sg:best_effort': 'false',
                                      'wrs-sg:group_size': '2'}
            }

        task.execute()

        sig_mock.assert_called_once_with(self.context, self.request_spec)
        task.scheduler_client.select_destinations.assert_called_once_with(
            self.context, clone_mock.return_value, [self.instance.uuid])
        prep_resize_mock.assert_called_once_with(
            self.context, self.instance, legacy_request_spec['image'],
            self.flavor, self.hosts[0]['host'], self.reservations,
            request_spec=legacy_request_spec_reloaded,
            filter_properties=self.filter_properties,
            node=self.hosts[0]['nodename'], clean_shutdown=self.clean_shutdown)
