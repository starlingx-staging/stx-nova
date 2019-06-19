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
import oslo_messaging as messaging
import six

from nova import availability_zones
from nova.compute import power_state
from nova.compute import utils as compute_utils
from nova.conductor.tasks import base
from nova.conductor.tasks import migrate
import nova.conf
from nova import exception
from nova.i18n import _
from nova import network
from nova import objects
from nova.objects import fields as obj_fields
from nova.scheduler import utils as scheduler_utils

LOG = logging.getLogger(__name__)
CONF = nova.conf.CONF


def supports_extended_port_binding(context, host):
    """Checks if the compute host service is new enough to support the neutron
    port binding-extended details.

    :param context: The user request context.
    :param host: The nova-compute host to check.
    :returns: True if the compute host is new enough to support extended
              port binding information, False otherwise.
    """
    svc = objects.Service.get_by_host_and_binary(context, host, 'nova-compute')
    return svc.version >= 35


class LiveMigrationTask(base.TaskBase):
    def __init__(self, context, instance, destination,
                 block_migration, disk_over_commit, migration, compute_rpcapi,
                 servicegroup_api, query_client, report_client,
                 request_spec=None):
        super(LiveMigrationTask, self).__init__(context, instance)
        self.destination = destination
        self.block_migration = block_migration
        self.disk_over_commit = disk_over_commit
        self.migration = migration
        self.source = instance.host
        self.migrate_data = None

        self.compute_rpcapi = compute_rpcapi
        self.servicegroup_api = servicegroup_api
        self.query_client = query_client
        self.report_client = report_client
        self.request_spec = request_spec
        self._source_cn = None
        self._held_allocations = None
        self.network_api = network.API()

    def _execute(self):
        self._check_instance_is_active()
        self._check_instance_has_no_numa()
        self._check_host_is_up(self.source)

        self._source_cn, self._held_allocations = (
            # NOTE(danms): This may raise various exceptions, which will
            # propagate to the API and cause a 500. This is what we
            # want, as it would indicate internal data structure corruption
            # (such as missing migrations, compute nodes, etc).
            migrate.replace_allocation_with_migration(self.context,
                                                      self.instance,
                                                      self.migration))

        if not self.destination:
            # Either no host was specified in the API request and the user
            # wants the scheduler to pick a destination host, or a host was
            # specified but is not forcing it, so they want the scheduler
            # filters to run on the specified host, like a scheduler hint.
            self.destination, dest_node = self._find_destination()
        else:
            # This is the case that the user specified the 'force' flag when
            # live migrating with a specific destination host so the scheduler
            # is bypassed. There are still some minimal checks performed here
            # though.
            source_node, dest_node = self._check_requested_destination()
            # Now that we're semi-confident in the force specified host, we
            # need to copy the source compute node allocations in Placement
            # to the destination compute node. Normally select_destinations()
            # in the scheduler would do this for us, but when forcing the
            # target host we don't call the scheduler.
            # TODO(mriedem): In Queens, call select_destinations() with a
            # skip_filters=True flag so the scheduler does the work of claiming
            # resources on the destination in Placement but still bypass the
            # scheduler filters, which honors the 'force' flag in the API.
            # This raises NoValidHost which will be handled in
            # ComputeTaskManager.
            # NOTE(gibi): consumer_generation = None as we expect that the
            # source host allocation is held by the migration therefore the
            # instance is a new, empty consumer for the dest allocation. If
            # this assumption fails then placement will return consumer
            # generation conflict and this call raise a AllocationUpdateFailed
            # exception. We let that propagate here to abort the migration.
            scheduler_utils.claim_resources_on_destination(
                self.context, self.report_client,
                self.instance, source_node, dest_node,
                source_allocations=self._held_allocations,
                consumer_generation=None)

            # dest_node is a ComputeNode object, so we need to get the actual
            # node name off it to set in the Migration object below.
            dest_node = dest_node.hypervisor_hostname

        self.instance.availability_zone = (
            availability_zones.get_host_availability_zone(
                self.context, self.destination))

        self.migration.source_node = self.instance.node
        self.migration.dest_node = dest_node
        self.migration.dest_compute = self.destination
        self.migration.save()

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
        if self._held_allocations:
            migrate.revert_allocation_for_migration(self.context,
                                                    self._source_cn,
                                                    self.instance,
                                                    self.migration)

    def _check_instance_is_active(self):
        if self.instance.power_state not in (power_state.RUNNING,
                                             power_state.PAUSED):
            raise exception.InstanceInvalidState(
                    instance_uuid=self.instance.uuid,
                    attr='power_state',
                    state=power_state.STATE_MAP[self.instance.power_state],
                    method='live migrate')

    def _check_instance_has_no_numa(self):
        """Prevent live migrations of instances with NUMA topologies."""
        if not self.instance.numa_topology:
            return

        # Only KVM (libvirt) supports NUMA topologies with CPU pinning;
        # HyperV's vNUMA feature doesn't allow specific pinning
        hypervisor_type = objects.ComputeNode.get_by_host_and_nodename(
            self.context, self.source, self.instance.node).hypervisor_type

        # KVM is not a hypervisor, so when using a virt_type of "kvm" the
        # hypervisor_type will still be "QEMU".
        if hypervisor_type.lower() != obj_fields.HVType.QEMU:
            return

        msg = ('Instance has an associated NUMA topology. '
               'Instance NUMA topologies, including related attributes '
               'such as CPU pinning, huge page and emulator thread '
               'pinning information, are not currently recalculated on '
               'live migration. See bug #1289064 for more information.'
               )

        if CONF.workarounds.enable_numa_live_migration:
            LOG.warning(msg, instance=self.instance)
        else:
            raise exception.MigrationPreCheckError(reason=msg)

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
        self._call_livem_checks_on_host(self.destination)
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

    def _call_livem_checks_on_host(self, destination):
        try:
            self.migrate_data = self.compute_rpcapi.\
                check_can_live_migrate_destination(self.context, self.instance,
                    destination, self.block_migration, self.disk_over_commit,
                    None, None)
        except messaging.MessagingTimeout:
            msg = _("Timeout while checking if we can live migrate to host: "
                    "%s") % destination
            raise exception.MigrationPreCheckError(msg)

        # Check to see that neutron supports the binding-extended API and
        # check to see that both the source and destination compute hosts
        # are new enough to support the new port binding flow.
        if (self.network_api.supports_port_binding_extension(self.context) and
                supports_extended_port_binding(self.context, self.source) and
                supports_extended_port_binding(self.context, destination)):
            self.migrate_data.vifs = (
                self._bind_ports_on_destination(destination))

    def _bind_ports_on_destination(self, destination):
        LOG.debug('Start binding ports on destination host: %s', destination,
                  instance=self.instance)
        # Bind ports on the destination host; returns a dict, keyed by
        # port ID, of a new destination host port binding dict per port
        # that was bound. This information is then stuffed into the
        # migrate_data.
        try:
            bindings = self.network_api.bind_ports_to_host(
                self.context, self.instance, destination)
        except exception.PortBindingFailed as e:
            # Port binding failed for that host, try another one.
            raise exception.MigrationPreCheckError(
                reason=e.format_message())

        source_vif_map = {
            vif['id']: vif for vif in self.instance.get_network_info()
        }
        migrate_vifs = []
        for port_id, binding in bindings.items():
            migrate_vif = objects.VIFMigrateData(
                port_id=port_id, **binding)
            migrate_vif.source_vif = source_vif_map[port_id]
            migrate_vifs.append(migrate_vif)
        return migrate_vifs

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

    def _get_request_spec_for_select_destinations(self, attempted_hosts=None):
        """Builds a RequestSpec that can be passed to select_destinations

        Used when calling the scheduler to pick a destination host for live
        migrating the instance.

        :param attempted_hosts: List of host names to ignore in the scheduler.
            This is generally at least seeded with the source host.
        :returns: nova.objects.RequestSpec object
        """
        request_spec = self.request_spec
        # NOTE(sbauza): Force_hosts/nodes needs to be reset
        # if we want to make sure that the next destination
        # is not forced to be the original host
        request_spec.reset_forced_destinations()

        # TODO(gibi): We need to make sure that the requested_resources field
        # is re calculated based on neutron ports.
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

        request_spec.ensure_project_and_user_id(self.instance)
        request_spec.ensure_network_metadata(self.instance)
        compute_utils.heal_reqspec_is_bfv(
            self.context, request_spec, self.instance)

        return request_spec

    def _find_destination(self):
        # TODO(johngarbutt) this retry loop should be shared
        attempted_hosts = [self.source]
        request_spec = self._get_request_spec_for_select_destinations(
            attempted_hosts)

        host = None
        while host is None:
            self._check_not_over_max_retries(attempted_hosts)
            request_spec.ignore_hosts = attempted_hosts
            try:
                selection_lists = self.query_client.select_destinations(
                        self.context, request_spec, [self.instance.uuid],
                        return_objects=True, return_alternates=False)
                # We only need the first item in the first list, as there is
                # only one instance, and we don't care about any alternates.
                selection = selection_lists[0][0]
                host = selection.service_host
            except messaging.RemoteError as ex:
                # TODO(ShaoHe Feng) There maybe multi-scheduler, and the
                # scheduling algorithm is R-R, we can let other scheduler try.
                # Note(ShaoHe Feng) There are types of RemoteError, such as
                # NoSuchMethod, UnsupportedVersion, we can distinguish it by
                # ex.exc_type.
                raise exception.MigrationSchedulerRPCError(
                    reason=six.text_type(ex))
            try:
                self._check_compatible_with_source_hypervisor(host)
                self._call_livem_checks_on_host(host)
            except (exception.Invalid, exception.MigrationPreCheckError) as e:
                LOG.debug("Skipping host: %(host)s because: %(e)s",
                    {"host": host, "e": e})
                attempted_hosts.append(host)
                # The scheduler would have created allocations against the
                # selected destination host in Placement, so we need to remove
                # those before moving on.
                self._remove_host_allocations(host, selection.nodename)
                host = None
        return selection.service_host, selection.nodename

    def _remove_host_allocations(self, host, node):
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

        # Now remove the allocations for our instance against that node.
        # Note that this does not remove allocations against any other node
        # or shared resource provider, it's just undoing what the scheduler
        # allocated for the given (destination) node.
        self.report_client.remove_provider_tree_from_instance_allocation(
            self.context, self.instance.uuid, compute_node.uuid)

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
