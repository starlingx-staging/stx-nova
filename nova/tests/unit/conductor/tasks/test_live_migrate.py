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
import oslo_messaging as messaging
from oslo_utils.fixture import uuidsentinel as uuids
import six

from nova.compute import power_state
from nova.compute import rpcapi as compute_rpcapi
from nova.compute import vm_states
from nova.conductor.tasks import live_migrate
from nova import context as nova_context
from nova import exception
from nova.network import model as network_model
from nova import objects
from nova.scheduler.client import query
from nova.scheduler.client import report
from nova.scheduler import utils as scheduler_utils
from nova import servicegroup
from nova import test
from nova.tests.unit import fake_instance


fake_selection1 = objects.Selection(service_host="host1", nodename="node1",
        cell_uuid=uuids.cell)
fake_selection2 = objects.Selection(service_host="host2", nodename="node2",
        cell_uuid=uuids.cell)


class LiveMigrationTaskTestCase(test.NoDBTestCase):
    def setUp(self):
        super(LiveMigrationTaskTestCase, self).setUp()
        self.context = nova_context.get_admin_context()
        self.instance_host = "host"
        self.instance_uuid = uuids.instance
        self.instance_image = "image_ref"
        db_instance = fake_instance.fake_db_instance(
                host=self.instance_host,
                uuid=self.instance_uuid,
                power_state=power_state.RUNNING,
                vm_state = vm_states.ACTIVE,
                memory_mb=512,
                image_ref=self.instance_image)
        self.instance = objects.Instance._from_db_object(
                self.context, objects.Instance(), db_instance)
        self.instance.system_metadata = {'image_hw_disk_bus': 'scsi'}
        self.instance.numa_topology = None
        self.destination = "destination"
        self.block_migration = "bm"
        self.disk_over_commit = "doc"
        self.migration = objects.Migration()
        self.fake_spec = objects.RequestSpec()
        self._generate_task()

        _p = mock.patch('nova.compute.utils.heal_reqspec_is_bfv')
        self.heal_reqspec_is_bfv_mock = _p.start()
        self.addCleanup(_p.stop)

        _p = mock.patch('nova.objects.RequestSpec.ensure_network_metadata')
        self.ensure_network_metadata_mock = _p.start()
        self.addCleanup(_p.stop)

    def _generate_task(self):
        self.task = live_migrate.LiveMigrationTask(self.context,
            self.instance, self.destination, self.block_migration,
            self.disk_over_commit, self.migration, compute_rpcapi.ComputeAPI(),
            servicegroup.API(), query.SchedulerQueryClient(),
            report.SchedulerReportClient(), self.fake_spec)

    @mock.patch('nova.availability_zones.get_host_availability_zone',
                return_value='fake-az')
    def test_execute_with_destination(self, mock_get_az):
        dest_node = objects.ComputeNode(hypervisor_hostname='dest_node')
        with test.nested(
            mock.patch.object(self.task, '_check_host_is_up'),
            mock.patch.object(self.task, '_check_requested_destination',
                              return_value=(mock.sentinel.source_node,
                                            dest_node)),
            mock.patch.object(scheduler_utils,
                              'claim_resources_on_destination'),
            mock.patch.object(self.migration, 'save'),
            mock.patch.object(self.task.compute_rpcapi, 'live_migration'),
            mock.patch('nova.conductor.tasks.migrate.'
                       'replace_allocation_with_migration'),
        ) as (mock_check_up, mock_check_dest, mock_claim, mock_save, mock_mig,
              m_alloc):
            mock_mig.return_value = "bob"
            m_alloc.return_value = (mock.MagicMock(), mock.sentinel.allocs)

            self.assertEqual("bob", self.task.execute())
            mock_check_up.assert_called_once_with(self.instance_host)
            mock_check_dest.assert_called_once_with()
            allocs = mock.sentinel.allocs
            mock_claim.assert_called_once_with(
                self.context, self.task.report_client,
                self.instance, mock.sentinel.source_node, dest_node,
                source_allocations=allocs, consumer_generation=None)
            mock_mig.assert_called_once_with(
                self.context,
                host=self.instance_host,
                instance=self.instance,
                dest=self.destination,
                block_migration=self.block_migration,
                migration=self.migration,
                migrate_data=None)
            self.assertTrue(mock_save.called)
            mock_get_az.assert_called_once_with(self.context, self.destination)
            self.assertEqual('fake-az', self.instance.availability_zone)
            # make sure the source/dest fields were set on the migration object
            self.assertEqual(self.instance.node, self.migration.source_node)
            self.assertEqual(dest_node.hypervisor_hostname,
                             self.migration.dest_node)
            self.assertEqual(self.task.destination,
                             self.migration.dest_compute)
            m_alloc.assert_called_once_with(self.context,
                                            self.instance,
                                            self.migration)
        # When the task is executed with a destination it means the host is
        # being forced and we don't call the scheduler, so we don't need to
        # heal the request spec.
        self.heal_reqspec_is_bfv_mock.assert_not_called()

        # When the task is executed with a destination it means the host is
        # being forced and we don't call the scheduler, so we don't need to
        # modify the request spec
        self.ensure_network_metadata_mock.assert_not_called()

    @mock.patch('nova.availability_zones.get_host_availability_zone',
                return_value='nova')
    def test_execute_without_destination(self, mock_get_az):
        self.destination = None
        self._generate_task()
        self.assertIsNone(self.task.destination)

        with test.nested(
            mock.patch.object(self.task, '_check_host_is_up'),
            mock.patch.object(self.task, '_find_destination'),
            mock.patch.object(self.task.compute_rpcapi, 'live_migration'),
            mock.patch.object(self.migration, 'save'),
            mock.patch('nova.conductor.tasks.migrate.'
                       'replace_allocation_with_migration'),
        ) as (mock_check, mock_find, mock_mig, mock_save, mock_alloc):
            mock_find.return_value = ("found_host", "found_node", None)
            mock_mig.return_value = "bob"
            mock_alloc.return_value = (mock.MagicMock(), mock.MagicMock())

            self.assertEqual("bob", self.task.execute())
            mock_check.assert_called_once_with(self.instance_host)
            mock_find.assert_called_once_with()
            mock_mig.assert_called_once_with(self.context,
                host=self.instance_host,
                instance=self.instance,
                dest="found_host",
                block_migration=self.block_migration,
                migration=self.migration,
                migrate_data=None)
            self.assertTrue(mock_save.called)
            mock_get_az.assert_called_once_with(self.context, 'found_host')
            self.assertEqual('found_host', self.migration.dest_compute)
            self.assertEqual('found_node', self.migration.dest_node)
            self.assertEqual(self.instance.node, self.migration.source_node)
            self.assertTrue(mock_alloc.called)

    def test_check_instance_is_active_passes_when_paused(self):
        self.task.instance['power_state'] = power_state.PAUSED
        self.task._check_instance_is_active()

    def test_check_instance_is_active_fails_when_shutdown(self):
        self.task.instance['power_state'] = power_state.SHUTDOWN
        self.assertRaises(exception.InstanceInvalidState,
                          self.task._check_instance_is_active)

    @mock.patch.object(objects.ComputeNode, 'get_by_host_and_nodename')
    def test_check_instance_has_no_numa_passes_no_numa(self, mock_get):
        self.flags(enable_numa_live_migration=False, group='workarounds')
        self.task.instance.numa_topology = None
        mock_get.return_value = objects.ComputeNode(
            uuid=uuids.cn1, hypervisor_type='kvm')
        self.task._check_instance_has_no_numa(self.context)

    @mock.patch.object(objects.ComputeNode, 'get_by_host_and_nodename')
    def test_check_instance_has_no_numa_passes_non_kvm(self, mock_get):
        self.flags(enable_numa_live_migration=False, group='workarounds')
        self.task.instance.numa_topology = objects.InstanceNUMATopology(
            cells=[objects.InstanceNUMACell(id=0, cpuset=set([0]),
                                            memory=1024)])
        mock_get.return_value = objects.ComputeNode(
            uuid=uuids.cn1, hypervisor_type='xen')
        self.task._check_instance_has_no_numa(self.context)

    @mock.patch.object(objects.Service, 'get_minimum_version',
                       return_value=38)
    @mock.patch.object(objects.ComputeNode, 'get_by_host_and_nodename')
    def test_check_instance_has_no_numa_passes_workaround(self, mock_get, _):
        self.flags(enable_numa_live_migration=True, group='workarounds')
        self.task.instance.numa_topology = objects.InstanceNUMATopology(
            cells=[objects.InstanceNUMACell(id=0, cpuset=set([0]),
                                            memory=1024)])
        mock_get.return_value = objects.ComputeNode(
            uuid=uuids.cn1, hypervisor_type='kvm')
        self.task._check_instance_has_no_numa(self.context)

    @mock.patch.object(objects.Service, 'get_minimum_version',
                       return_value=38)
    @mock.patch.object(objects.ComputeNode, 'get_by_host_and_nodename')
    def test_check_instance_has_no_numa_fails(self, mock_get, _):
        self.flags(enable_numa_live_migration=False, group='workarounds')
        mock_get.return_value = objects.ComputeNode(
            uuid=uuids.cn1, hypervisor_type='QEMU')
        self.task.instance.numa_topology = objects.InstanceNUMATopology(
            cells=[objects.InstanceNUMACell(id=0, cpuset=set([0]),
                                            memory=1024)])
        self.assertRaises(exception.MigrationPreCheckError,
                          self.task._check_instance_has_no_numa,
                          self.context)

    @mock.patch.object(objects.Service, 'get_minimum_version',
                       return_value=39)
    @mock.patch.object(objects.ComputeNode, 'get_by_host_and_nodename')
    def test_check_instance_has_no_numa_new_svc_passes(self, mock_get, _):
        self.flags(enable_numa_live_migration=False, group='workarounds')
        mock_get.return_value = objects.ComputeNode(
            uuid=uuids.cn1, hypervisor_type='kvm')
        self.task.instance.numa_topology = objects.InstanceNUMATopology(
            cells=[objects.InstanceNUMACell(id=0, cpuset=set([0]),
                                            memory=1024)])
        self.task._check_instance_has_no_numa(self.context)

    @mock.patch.object(objects.Service, 'get_by_compute_host')
    @mock.patch.object(servicegroup.API, 'service_is_up')
    def test_check_instance_host_is_up(self, mock_is_up, mock_get):
        mock_get.return_value = "service"
        mock_is_up.return_value = True

        self.task._check_host_is_up("host")
        mock_get.assert_called_once_with(self.context, "host")
        mock_is_up.assert_called_once_with("service")

    @mock.patch.object(objects.Service, 'get_by_compute_host')
    @mock.patch.object(servicegroup.API, 'service_is_up')
    def test_check_instance_host_is_up_fails_if_not_up(self, mock_is_up,
                                                       mock_get):
        mock_get.return_value = "service"
        mock_is_up.return_value = False

        self.assertRaises(exception.ComputeServiceUnavailable,
                self.task._check_host_is_up, "host")
        mock_get.assert_called_once_with(self.context, "host")
        mock_is_up.assert_called_once_with("service")

    @mock.patch.object(objects.Service, 'get_by_compute_host',
                       side_effect=exception.ComputeHostNotFound(host='host'))
    def test_check_instance_host_is_up_fails_if_not_found(self, mock):
        self.assertRaises(exception.ComputeHostNotFound,
            self.task._check_host_is_up, "host")

    @mock.patch.object(objects.Service, 'get_by_compute_host')
    @mock.patch.object(live_migrate.LiveMigrationTask, '_get_compute_info')
    @mock.patch.object(servicegroup.API, 'service_is_up')
    @mock.patch.object(compute_rpcapi.ComputeAPI,
                       'check_can_live_migrate_destination')
    def test_check_requested_destination(self, mock_check, mock_is_up,
                                         mock_get_info, mock_get_host):
        mock_get_host.return_value = "service"
        mock_is_up.return_value = True
        hypervisor_details = objects.ComputeNode(
            hypervisor_type="a",
            hypervisor_version=6.1,
            free_ram_mb=513,
            memory_mb=512,
            ram_allocation_ratio=1.0)
        mock_get_info.return_value = hypervisor_details
        mock_check.return_value = "migrate_data"

        with mock.patch.object(self.task.network_api,
                               'supports_port_binding_extension',
                               return_value=False):
            self.assertEqual((hypervisor_details, hypervisor_details),
                             self.task._check_requested_destination())
        self.assertEqual("migrate_data", self.task.migrate_data)
        mock_get_host.assert_called_once_with(self.context, self.destination)
        mock_is_up.assert_called_once_with("service")
        self.assertEqual([mock.call(self.destination),
                          mock.call(self.instance_host),
                          mock.call(self.destination)],
                         mock_get_info.call_args_list)
        mock_check.assert_called_once_with(self.context, self.instance,
            self.destination, self.block_migration, self.disk_over_commit,
            self.task.migration, None)

    def test_check_requested_destination_fails_with_same_dest(self):
        self.task.destination = "same"
        self.task.source = "same"
        self.assertRaises(exception.UnableToMigrateToSelf,
                          self.task._check_requested_destination)

    @mock.patch.object(objects.Service, 'get_by_compute_host',
                       side_effect=exception.ComputeHostNotFound(host='host'))
    def test_check_requested_destination_fails_when_destination_is_up(self,
                                                                      mock):
        self.assertRaises(exception.ComputeHostNotFound,
                          self.task._check_requested_destination)

    @mock.patch.object(live_migrate.LiveMigrationTask, '_check_host_is_up')
    @mock.patch.object(objects.ComputeNode,
                       'get_first_node_by_host_for_old_compat')
    def test_check_requested_destination_fails_with_not_enough_memory(
        self, mock_get_first, mock_is_up):
        mock_get_first.return_value = (
            objects.ComputeNode(free_ram_mb=513,
                                memory_mb=1024,
                                ram_allocation_ratio=0.9,))

        # free_ram is bigger than instance.ram (512) but the allocation
        # ratio reduces the total available RAM to 410MB
        # (1024 * 0.9 - (1024 - 513))
        self.assertRaises(exception.MigrationPreCheckError,
                          self.task._check_requested_destination)
        mock_is_up.assert_called_once_with(self.destination)
        mock_get_first.assert_called_once_with(self.context, self.destination)

    @mock.patch.object(live_migrate.LiveMigrationTask, '_check_host_is_up')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_destination_has_enough_memory')
    @mock.patch.object(live_migrate.LiveMigrationTask, '_get_compute_info')
    def test_check_requested_destination_fails_with_hypervisor_diff(
        self, mock_get_info, mock_check, mock_is_up):
        mock_get_info.side_effect = [
            objects.ComputeNode(hypervisor_type='b'),
            objects.ComputeNode(hypervisor_type='a')]

        self.assertRaises(exception.InvalidHypervisorType,
                          self.task._check_requested_destination)
        mock_is_up.assert_called_once_with(self.destination)
        mock_check.assert_called_once_with()
        self.assertEqual([mock.call(self.instance_host),
                          mock.call(self.destination)],
                         mock_get_info.call_args_list)

    @mock.patch.object(live_migrate.LiveMigrationTask, '_check_host_is_up')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_destination_has_enough_memory')
    @mock.patch.object(live_migrate.LiveMigrationTask, '_get_compute_info')
    def test_check_requested_destination_fails_with_hypervisor_too_old(
        self, mock_get_info, mock_check, mock_is_up):
        host1 = {'hypervisor_type': 'a', 'hypervisor_version': 7}
        host2 = {'hypervisor_type': 'a', 'hypervisor_version': 6}
        mock_get_info.side_effect = [objects.ComputeNode(**host1),
                                     objects.ComputeNode(**host2)]

        self.assertRaises(exception.DestinationHypervisorTooOld,
                          self.task._check_requested_destination)
        mock_is_up.assert_called_once_with(self.destination)
        mock_check.assert_called_once_with()
        self.assertEqual([mock.call(self.instance_host),
                          mock.call(self.destination)],
                         mock_get_info.call_args_list)

    @mock.patch.object(objects.Service, 'get_by_compute_host')
    @mock.patch.object(live_migrate.LiveMigrationTask, '_get_compute_info')
    @mock.patch.object(servicegroup.API, 'service_is_up')
    @mock.patch.object(compute_rpcapi.ComputeAPI,
                       'check_can_live_migrate_destination')
    @mock.patch.object(objects.HostMapping, 'get_by_host',
                       return_value=objects.HostMapping(
                           cell_mapping=objects.CellMapping(
                               uuid=uuids.different)))
    def test_check_requested_destination_fails_different_cells(
            self, mock_get_host_mapping, mock_check, mock_is_up,
            mock_get_info, mock_get_host):
        mock_get_host.return_value = "service"
        mock_is_up.return_value = True
        hypervisor_details = objects.ComputeNode(
            hypervisor_type="a",
            hypervisor_version=6.1,
            free_ram_mb=513,
            memory_mb=512,
            ram_allocation_ratio=1.0)
        mock_get_info.return_value = hypervisor_details
        mock_check.return_value = "migrate_data"

        with mock.patch.object(self.task.network_api,
                               'supports_port_binding_extension',
                               return_value=False):
            ex = self.assertRaises(exception.MigrationPreCheckError,
                                   self.task._check_requested_destination)
        self.assertIn('across cells', six.text_type(ex))

    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_call_livem_checks_on_host')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_compatible_with_source_hypervisor')
    @mock.patch.object(query.SchedulerQueryClient, 'select_destinations',
                       return_value=[[fake_selection1]])
    @mock.patch.object(objects.RequestSpec, 'reset_forced_destinations')
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    def test_find_destination_works(self, mock_setup, mock_reset, mock_select,
                                    mock_check, mock_call):
        self.assertEqual(("host1", "node1", None),
                         self.task._find_destination())

        # Make sure the request_spec was updated to include the cell
        # mapping.
        self.assertIsNotNone(self.fake_spec.requested_destination.cell)
        # Make sure the spec was updated to include the project_id.
        self.assertEqual(self.fake_spec.project_id, self.instance.project_id)

        mock_setup.assert_called_once_with(self.context, self.fake_spec)
        mock_reset.assert_called_once_with()
        self.ensure_network_metadata_mock.assert_called_once_with(
            self.instance)
        self.heal_reqspec_is_bfv_mock.assert_called_once_with(
            self.context, self.fake_spec, self.instance)
        mock_select.assert_called_once_with(self.context, self.fake_spec,
            [self.instance.uuid], return_objects=True, return_alternates=False)
        mock_check.assert_called_once_with('host1')
        mock_call.assert_called_once_with('host1')

    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_call_livem_checks_on_host')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_compatible_with_source_hypervisor')
    @mock.patch.object(query.SchedulerQueryClient, 'select_destinations',
                       return_value=[[fake_selection1]])
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    def test_find_destination_no_image_works(self, mock_setup, mock_select,
                                             mock_check, mock_call):
        self.instance['image_ref'] = ''

        self.assertEqual(("host1", "node1", None),
                         self.task._find_destination())

        mock_setup.assert_called_once_with(self.context, self.fake_spec)
        mock_select.assert_called_once_with(
            self.context, self.fake_spec, [self.instance.uuid],
            return_objects=True, return_alternates=False)
        mock_check.assert_called_once_with('host1')
        mock_call.assert_called_once_with('host1')

    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_remove_host_allocations')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_call_livem_checks_on_host')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_compatible_with_source_hypervisor')
    @mock.patch.object(query.SchedulerQueryClient, 'select_destinations',
                       side_effect=[[[fake_selection1]], [[fake_selection2]]])
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    def _test_find_destination_retry_hypervisor_raises(
            self, error, mock_setup, mock_select, mock_check, mock_call,
            mock_remove):
        mock_check.side_effect = [error, None]

        self.assertEqual(("host2", "node2", None),
                         self.task._find_destination())
        # Should have removed allocations for the first host.
        mock_remove.assert_called_once_with('host1', 'node1')
        mock_setup.assert_called_once_with(self.context, self.fake_spec)
        mock_select.assert_has_calls([
            mock.call(self.context, self.fake_spec, [self.instance.uuid],
                      return_objects=True, return_alternates=False),
            mock.call(self.context, self.fake_spec, [self.instance.uuid],
                      return_objects=True, return_alternates=False)])
        mock_check.assert_has_calls([mock.call('host1'), mock.call('host2')])
        mock_call.assert_called_once_with('host2')

    def test_find_destination_retry_with_old_hypervisor(self):
        self._test_find_destination_retry_hypervisor_raises(
                exception.DestinationHypervisorTooOld)

    def test_find_destination_retry_with_invalid_hypervisor_type(self):
        self._test_find_destination_retry_hypervisor_raises(
                exception.InvalidHypervisorType)

    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_remove_host_allocations')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_call_livem_checks_on_host')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_compatible_with_source_hypervisor')
    @mock.patch.object(query.SchedulerQueryClient, 'select_destinations',
                       side_effect=[[[fake_selection1]], [[fake_selection2]]])
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    def test_find_destination_retry_with_invalid_livem_checks(
            self, mock_setup, mock_select, mock_check, mock_call, mock_remove):
        self.flags(migrate_max_retries=1)
        mock_call.side_effect = [exception.Invalid(), None]

        self.assertEqual(("host2", "node2", None),
                         self.task._find_destination())
        # Should have removed allocations for the first host.
        mock_remove.assert_called_once_with('host1', 'node1')
        mock_setup.assert_called_once_with(self.context, self.fake_spec)
        mock_select.assert_has_calls([
            mock.call(self.context, self.fake_spec, [self.instance.uuid],
                      return_objects=True, return_alternates=False),
            mock.call(self.context, self.fake_spec, [self.instance.uuid],
                      return_objects=True, return_alternates=False)])
        mock_check.assert_has_calls([mock.call('host1'), mock.call('host2')])
        mock_call.assert_has_calls([mock.call('host1'), mock.call('host2')])

    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_remove_host_allocations')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_call_livem_checks_on_host')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_compatible_with_source_hypervisor')
    @mock.patch.object(query.SchedulerQueryClient, 'select_destinations',
                       side_effect=[[[fake_selection1]], [[fake_selection2]]])
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    def test_find_destination_retry_with_failed_migration_pre_checks(
            self, mock_setup, mock_select, mock_check, mock_call, mock_remove):
        self.flags(migrate_max_retries=1)
        mock_call.side_effect = [exception.MigrationPreCheckError('reason'),
                                 None]

        self.assertEqual(("host2", "node2", None),
                         self.task._find_destination())
        # Should have removed allocations for the first host.
        mock_remove.assert_called_once_with('host1', 'node1')
        mock_setup.assert_called_once_with(self.context, self.fake_spec)
        mock_select.assert_has_calls([
            mock.call(self.context, self.fake_spec, [self.instance.uuid],
                      return_objects=True, return_alternates=False),
            mock.call(self.context, self.fake_spec, [self.instance.uuid],
                      return_objects=True, return_alternates=False)])
        mock_check.assert_has_calls([mock.call('host1'), mock.call('host2')])
        mock_call.assert_has_calls([mock.call('host1'), mock.call('host2')])

    @mock.patch.object(objects.Migration, 'save')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_remove_host_allocations')
    @mock.patch.object(live_migrate.LiveMigrationTask,
                       '_check_compatible_with_source_hypervisor',
                       side_effect=exception.DestinationHypervisorTooOld())
    @mock.patch.object(query.SchedulerQueryClient, 'select_destinations',
                       return_value=[[fake_selection1]])
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    def test_find_destination_retry_exceeds_max(
            self, mock_setup, mock_select, mock_check, mock_remove, mock_save):
        self.flags(migrate_max_retries=0)

        self.assertRaises(exception.MaxRetriesExceeded,
                          self.task._find_destination)
        self.assertEqual('failed', self.task.migration.status)
        mock_save.assert_called_once_with()
        # Should have removed allocations for the first host.
        mock_remove.assert_called_once_with('host1', 'node1')
        mock_setup.assert_called_once_with(self.context, self.fake_spec)
        mock_select.assert_called_once_with(
            self.context, self.fake_spec, [self.instance.uuid],
            return_objects=True, return_alternates=False)
        mock_check.assert_called_once_with('host1')

    @mock.patch.object(query.SchedulerQueryClient, 'select_destinations',
                       side_effect=exception.NoValidHost(reason=""))
    @mock.patch.object(scheduler_utils, 'setup_instance_group')
    def test_find_destination_when_runs_out_of_hosts(self, mock_setup,
                                                     mock_select):
        self.assertRaises(exception.NoValidHost, self.task._find_destination)
        mock_setup.assert_called_once_with(self.context, self.fake_spec)
        mock_select.assert_called_once_with(
            self.context, self.fake_spec, [self.instance.uuid],
            return_objects=True, return_alternates=False)

    @mock.patch("nova.utils.get_image_from_system_metadata")
    @mock.patch("nova.scheduler.utils.build_request_spec")
    @mock.patch("nova.scheduler.utils.setup_instance_group")
    @mock.patch("nova.objects.RequestSpec.from_primitives")
    def test_find_destination_with_remoteError(self,
        m_from_primitives, m_setup_instance_group,
        m_build_request_spec, m_get_image_from_system_metadata):
        m_get_image_from_system_metadata.return_value = {'properties': {}}
        m_build_request_spec.return_value = {}
        fake_spec = objects.RequestSpec()
        m_from_primitives.return_value = fake_spec
        with mock.patch.object(self.task.query_client,
            'select_destinations') as m_select_destinations:
            error = messaging.RemoteError()
            m_select_destinations.side_effect = error
            self.assertRaises(exception.MigrationSchedulerRPCError,
                              self.task._find_destination)

    def test_call_livem_checks_on_host(self):
        with mock.patch.object(self.task.compute_rpcapi,
            'check_can_live_migrate_destination',
            side_effect=messaging.MessagingTimeout):
            self.assertRaises(exception.MigrationPreCheckError,
                self.task._call_livem_checks_on_host, {})

    @mock.patch('nova.conductor.tasks.live_migrate.'
                'supports_extended_port_binding', return_value=True)
    def test_call_livem_checks_on_host_bind_ports(self, mock_supports_ext):
        data = objects.LibvirtLiveMigrateData()
        bindings = {
            uuids.port1: {'host': 'dest-host'},
            uuids.port2: {'host': 'dest-host'}
        }

        @mock.patch.object(self.task.compute_rpcapi,
                           'check_can_live_migrate_destination',
                           return_value=data)
        @mock.patch.object(self.task.network_api,
                           'supports_port_binding_extension',
                           return_value=True)
        @mock.patch.object(self.task.network_api,
                           'bind_ports_to_host', return_value=bindings)
        def _test(mock_bind_ports_to_host, mock_supports_port_binding,
                  mock_check_can_live_migrate_dest):
            nwinfo = network_model.NetworkInfo([
                network_model.VIF(uuids.port1),
                network_model.VIF(uuids.port2)])
            self.instance.info_cache = objects.InstanceInfoCache(
                network_info=nwinfo)
            self.task._call_livem_checks_on_host('dest-host')
            # Assert the migrate_data set on the task based on the port
            # bindings created.
            self.assertIn('vifs', data)
            self.assertEqual(2, len(data.vifs))
            for vif in data.vifs:
                self.assertIn('source_vif', vif)
                self.assertEqual('dest-host', vif.host)
                self.assertEqual(vif.port_id, vif.source_vif['id'])

        _test()

    @mock.patch.object(objects.InstanceMapping, 'get_by_instance_uuid',
                       side_effect=exception.InstanceMappingNotFound(
                           uuid=uuids.instance))
    def test_get_source_cell_mapping_not_found(self, mock_get):
        """Negative test where InstanceMappingNotFound is raised and converted
        to MigrationPreCheckError.
        """
        self.assertRaises(exception.MigrationPreCheckError,
                          self.task._get_source_cell_mapping)
        mock_get.assert_called_once_with(
            self.task.context, self.task.instance.uuid)

    @mock.patch.object(objects.HostMapping, 'get_by_host',
                       side_effect=exception.HostMappingNotFound(
                           name='destination'))
    def test_get_destination_cell_mapping_not_found(self, mock_get):
        """Negative test where HostMappingNotFound is raised and converted
        to MigrationPreCheckError.
        """
        self.assertRaises(exception.MigrationPreCheckError,
                          self.task._get_destination_cell_mapping)
        mock_get.assert_called_once_with(
            self.task.context, self.task.destination)

    @mock.patch.object(objects.ComputeNode, 'get_by_host_and_nodename',
                       side_effect=exception.ComputeHostNotFound(host='host'))
    @mock.patch('nova.scheduler.client.report.SchedulerReportClient.'
                'remove_provider_tree_from_instance_allocation')
    def test_remove_host_allocations_compute_host_not_found(
            self, remove_provider, get_cn):
        """Tests that failing to find a ComputeNode will not blow up
        the _remove_host_allocations method.
        """
        self.task._remove_host_allocations('host', 'node')
        remove_provider.assert_not_called()
