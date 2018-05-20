# Copyright (c) 2012 OpenStack Foundation
# All Rights Reserved.
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
# Copyright (c) 2014-2017 Wind River Systems, Inc.
#

"""
Track resources like memory and disk for a compute host.  Provides the
scheduler with useful information about availability through the ComputeNode
model.
"""
import collections
import copy
import math
import sys

from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import units

from nova.compute import claims
from nova.compute import monitors
from nova.compute import power_state
from nova.compute import stats
from nova.compute import task_states
from nova.compute import utils as compute_utils
from nova.compute import vm_states
import nova.conf
import nova.context
from nova import exception
from nova.i18n import _
from nova import objects
from nova.objects import base as obj_base
from nova.objects import fields
from nova.objects import migration as migration_obj
from nova.pci import manager as pci_manager
from nova.pci import request as pci_request
from nova.pci import utils as pci_utils
from nova import rpc
from nova.scheduler import client as scheduler_client
from nova.scheduler import utils as scheduler_utils
from nova import utils
from nova.virt import hardware

CONF = nova.conf.CONF

LOG = logging.getLogger(__name__)
COMPUTE_RESOURCE_SEMAPHORE = "compute_resources"

# WRS: tracker debug logging
_usage_debug = dict()


# WRS: migration tracking
def _instance_in_migration_or_resize_state(instance):
    """Returns True if the instance is in progress of migrating or resizing.

    :param instance: `nova.objects.Instance` object
    """
    vm = instance.vm_state
    task = instance.task_state

    if vm == vm_states.RESIZED:
        return True

    if (vm in [vm_states.ACTIVE, vm_states.STOPPED]
            and task in [task_states.RESIZE_PREP,
            task_states.RESIZE_MIGRATING, task_states.RESIZE_MIGRATED,
            task_states.RESIZE_FINISH, task_states.REBUILDING,
            task_states.MIGRATING]):
        return True

    # WRS: handle evacuation case where instance is in ERROR state
    if (vm == vm_states.ERROR and task in [task_states.REBUILDING,
            task_states.REBUILD_BLOCK_DEVICE_MAPPING,
            task_states.REBUILD_SPAWNING]):
        return True

    return False


def _normalize_inventory_from_cn_obj(inv_data, cn):
    """Helper function that injects various information from a compute node
    object into the inventory dict returned from the virt driver's
    get_inventory() method. This function allows us to marry information like
    *_allocation_ratio and reserved memory amounts that are in the
    compute_nodes DB table and that the virt driver doesn't know about with the
    information the virt driver *does* know about.

    Note that if the supplied inv_data contains allocation_ratio, reserved or
    other fields, we DO NOT override the value with that of the compute node.
    This is to ensure that the virt driver is the single source of truth
    regarding inventory information. For instance, the Ironic virt driver will
    always return a very specific inventory with allocation_ratios pinned to
    1.0.

    :param inv_data: Dict, keyed by resource class, of inventory information
                     returned from virt driver's get_inventory() method
    :param compute_node: `objects.ComputeNode` describing the compute node
    """
    if fields.ResourceClass.VCPU in inv_data:
        cpu_inv = inv_data[fields.ResourceClass.VCPU]
        if 'allocation_ratio' not in cpu_inv:
            # WRS: for libvirt driver adjust vcpus by allocation ratio. Leave
            # the rest (e.g. ironic) alone
            cpu_inv['total'] = int(cpu_inv['total'] * cn.cpu_allocation_ratio)
            cpu_inv['max_unit'] = int(cpu_inv['max_unit'] *
                                                      cn.cpu_allocation_ratio)
            cpu_inv['allocation_ratio'] = 1
        if 'reserved' not in cpu_inv:
            cpu_inv['reserved'] = CONF.reserved_host_cpus

    if fields.ResourceClass.MEMORY_MB in inv_data:
        mem_inv = inv_data[fields.ResourceClass.MEMORY_MB]
        if 'allocation_ratio' not in mem_inv:
            mem_inv['allocation_ratio'] = cn.ram_allocation_ratio
        if 'reserved' not in mem_inv:
            mem_inv['reserved'] = CONF.reserved_host_memory_mb

    if fields.ResourceClass.DISK_GB in inv_data:
        disk_inv = inv_data[fields.ResourceClass.DISK_GB]
        if 'allocation_ratio' not in disk_inv:
            disk_inv['allocation_ratio'] = cn.disk_allocation_ratio
        if 'reserved' not in disk_inv:
            # TODO(johngarbutt) We should either move to reserved_host_disk_gb
            # or start tracking DISK_MB.
            reserved_mb = CONF.reserved_host_disk_mb
            reserved_gb = compute_utils.convert_mb_to_ceil_gb(reserved_mb)
            disk_inv['reserved'] = reserved_gb


class ResourceTracker(object):
    """Compute helper class for keeping track of resource usage as instances
    are built and destroyed.
    """

    def __init__(self, host, driver):
        self.host = host
        self.driver = driver
        self.pci_tracker = None
        # Dict of objects.ComputeNode objects, keyed by nodename
        self.compute_nodes = {}
        self.stats = stats.Stats()
        self.tracked_instances = {}
        self.tracked_migrations = {}
        monitor_handler = monitors.MonitorHandler(self)
        self.monitors = monitor_handler.monitors
        self.old_resources = collections.defaultdict(objects.ComputeNode)
        self.scheduler_client = scheduler_client.SchedulerClient()
        self.reportclient = self.scheduler_client.reportclient
        self.ram_allocation_ratio = CONF.ram_allocation_ratio
        self.cpu_allocation_ratio = CONF.cpu_allocation_ratio
        self.disk_allocation_ratio = CONF.disk_allocation_ratio
        self.tracked_in_progress = []
        self.cached_dal = (self.driver.get_disk_available_least()
                           if driver else 0)

    def _get_compat_cpu(self, instance, resources):
        """Helper function to reserve a pcpu from the host

        Search for an unpinned cpu on the same numa node as what the instance
        is already using.  Use node of vcpu0 for now, will have to get fancier
        to deal with shared platform CPU, HT siblings, and multi-numa-node
        instances.
        """
        # Make sure we don't squeeze out globally floating instances.
        # We don't currently support this, but we might eventually.
        if (resources.vcpus - resources.vcpus_used) < 1:
            raise exception.ComputeResourcesUnavailable(
                        reason="no free pcpu available on host")

        vcpu0_cell, vcpu0_phys = hardware.instance_vcpu_to_pcpu(instance, 0)
        host_numa_topology, jsonify_result = \
            hardware.host_topology_and_format_from_host(resources)

        for cell in host_numa_topology.cells:
            if vcpu0_cell.id == cell.id:
                host_has_threads = (cell.siblings and
                                    len(cell.siblings[0]) > 1)
                if (host_has_threads and vcpu0_cell.cpu_thread_policy ==
                        fields.CPUThreadAllocationPolicy.ISOLATE):
                    # We need to allocate a set of matched sibling threads,
                    # and we don't know how many siblings are on a core.
                    # Also, cell.free_siblings can have empty sets or sets
                    # with not all siblings free, we need to filter them out.
                    num_sibs = len(cell.siblings[0])
                    free_siblings = [sibs for sibs in cell.free_siblings if
                                     len(sibs) == num_sibs]
                    if not free_siblings:
                        raise exception.ComputeResourcesUnavailable(
                            reason="no free siblings available on NUMA node")
                    try:
                        siblings = min(free_siblings)
                        pcpu = min(siblings)
                    except ValueError:
                        # Shouldn't happen given above check of free_siblings
                        raise exception.ComputeResourcesUnavailable(
                            reason="unable to find free sibling cpu "
                                   "on NUMA node")
                    # Update the host numa topology
                    cell.pin_cpus(siblings, strict=True)
                    cell.cpu_usage += len(siblings)
                    resources.vcpus_used += len(siblings)
                else:
                    if cell.avail_cpus < 1:
                        raise exception.ComputeResourcesUnavailable(
                            reason="no free pcpu available on NUMA node")
                    try:
                        pcpu = min(cell.free_cpus)
                    except ValueError:
                        # Shouldn't happen given the above check of avail_cpus
                        raise exception.ComputeResourcesUnavailable(
                            reason="unable to find free cpu on NUMA node")
                    # Update the host numa topology
                    cell.pin_cpus(set([pcpu]), strict=True)
                    cell.cpu_usage += 1
                    resources.vcpus_used += 1

                if jsonify_result:
                    resources.numa_topology = host_numa_topology._to_json()
                # WRS: update the affinity of instances with floating CPUs.
                hardware.update_floating_affinity(resources)
                return pcpu
        err = (_("Couldn't find host numa node containing pcpu %d") %
                 vcpu0_phys)
        raise exception.InternalError(message=err)

    def _put_compat_cpu(self, instance, pcpu, resources):
        """Helper function to return a pcpu back to the host """
        host_numa_topology, jsonify_result = \
            hardware.host_topology_and_format_from_host(resources)
        for cell in host_numa_topology.cells:
            if pcpu in cell.pinned_cpus:
                thread_policy_is_isolate = (
                    instance.numa_topology.cells[0].cpu_thread_policy ==
                    fields.CPUThreadAllocationPolicy.ISOLATE)
                # Don't assume we know how many siblings are on a core.
                host_has_threads = (cell.siblings and
                                    len(cell.siblings[0]) > 1)
                if (host_has_threads and thread_policy_is_isolate):
                    # WRS: non-strict pinning accounting when freeing
                    cell.unpin_cpus_with_siblings(set([pcpu]), strict=False)
                    cell.cpu_usage -= len(cell.siblings[0])
                    resources.vcpus_used -= len(cell.siblings[0])
                else:
                    # WRS: non-strict pinning accounting when freeing
                    cell.unpin_cpus(set([pcpu]), strict=False)
                    cell.cpu_usage -= 1
                    resources.vcpus_used -= 1
                if jsonify_result:
                    resources.numa_topology = host_numa_topology._to_json()
                # WRS: update the affinity of instances with floating CPUs.
                hardware.update_floating_affinity(resources)
                return
        err = (_("Couldn't find host numa node containing pcpu %d") % pcpu)
        raise exception.InternalError(message=err)

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def get_compat_cpu(self, instance, vcpu_cell, vcpu, nodename):
        # We hold the semaphore here so we don't race against the
        # resource audit.

        # Find pcpu and update the CPU usage on the host.
        cn = self.compute_nodes[nodename]
        pcpu = self._get_compat_cpu(instance, cn)

        # Update the instance CPU pinning to map the vcpu to the pcpu
        vcpu_cell.pin(vcpu, pcpu)
        instance.vcpus += 1
        instance.save()

        return pcpu

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def put_compat_cpu(self, instance, cpu, vcpu_cell, vcpu, vcpu0_phys,
                       nodename):
        # We hold the semaphore here so we don't race against the
        # resource audit.

        # Update the instance NUMA cell CPU pinning to map the vcpu
        # to the same pCPU as vcpu0.
        vcpu_cell.pin(vcpu, vcpu0_phys)
        instance.vcpus -= 1
        instance.save()

        # Now update the CPU usage on the host
        cn = self.compute_nodes[nodename]
        self._put_compat_cpu(instance, cpu, cn)

    def _copy_if_bfv(self, instance, instance_type=None):
        instance_copy = None
        instance_type_copy = None
        if instance.is_volume_backed():
            instance_copy = instance.obj_clone()
            instance_copy.flavor.root_gb = 0
            if instance_type is not None:
                instance_type_copy = instance_type.obj_clone()
                instance_type_copy.root_gb = 0
        if instance_copy is None:
            # If we didn't need to clone the object, use original
            instance_copy = instance
        if instance_type_copy is None:
            # If we didn't need to clone the object, use original
            instance_type_copy = instance_type
        return instance_copy, instance_type_copy

    def _copy_instance_type_if_bfv(self, instance, instance_type):
        instance_type_copy = None
        if instance.is_volume_backed():
            instance_type_copy = instance_type.obj_clone()
            instance_type_copy.root_gb = 0
        if instance_type_copy is None:
            # If we didn't need to clone the object, use original
            instance_type_copy = instance_type
        return instance_type_copy

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def instance_claim(self, context, instance, nodename, limits=None):
        """Indicate that some resources are needed for an upcoming compute
        instance build operation.

        This should be called before the compute node is about to perform
        an instance build operation that will consume additional resources.

        :param context: security context
        :param instance: instance to reserve resources for.
        :type instance: nova.objects.instance.Instance object
        :param nodename: The Ironic nodename selected by the scheduler
        :param limits: Dict of oversubscription limits for memory, disk,
                       and CPUs.
        :returns: A Claim ticket representing the reserved resources.  It can
                  be used to revert the resource usage if an error occurs
                  during the instance build.
        """
        if self.disabled(nodename):
            # instance_claim() was called before update_available_resource()
            # (which ensures that a compute node exists for nodename). We
            # shouldn't get here but in case we do, just set the instance's
            # host and nodename attribute (probably incorrect) and return a
            # NoopClaim.
            # TODO(jaypipes): Remove all the disabled junk from the resource
            # tracker. Servicegroup API-level active-checking belongs in the
            # nova-compute manager.
            self._set_instance_host_and_node(instance, nodename)
            return claims.NopClaim()

        # sanity checks:
        if instance.host:
            LOG.warning("Host field should not be set on the instance "
                        "until resources have been claimed.",
                        instance=instance)

        if instance.node:
            LOG.warning("Node field should not be set on the instance "
                        "until resources have been claimed.",
                        instance=instance)

        # TODO(melwitt/jaypipes): Remove this after resource-providers can
        # handle claims and reporting for boot-from-volume.
        inst_copy_zero_disk, _ = self._copy_if_bfv(instance)

        # get the overhead required to build this instance:
        overhead = self.driver.estimate_instance_overhead(inst_copy_zero_disk)
        LOG.debug("Memory overhead for %(flavor)d MB instance; %(overhead)d "
                  "MB", {'flavor': inst_copy_zero_disk.flavor.memory_mb,
                          'overhead': overhead['memory_mb']})
        LOG.debug("Disk overhead for %(flavor)d GB instance; %(overhead)d "
                  "GB", {'flavor': inst_copy_zero_disk.flavor.root_gb,
                         'overhead': overhead.get('disk_gb', 0)})
        LOG.debug("CPU overhead for %(flavor)d vCPUs instance; %(overhead)d "
                  "vCPU(s)", {'flavor': inst_copy_zero_disk.flavor.vcpus,
                              'overhead': overhead.get('vcpus', 0)})

        cn = self.compute_nodes[nodename]
        pci_requests = objects.InstancePCIRequests.get_by_instance_uuid(
            context, instance.uuid)
        claim = claims.Claim(context, inst_copy_zero_disk, nodename, self, cn,
                             pci_requests, overhead=overhead, limits=limits)

        # self._set_instance_host_and_node() will save instance to the DB
        # so set instance.numa_topology first.  We need to make sure
        # that numa_topology is saved while under COMPUTE_RESOURCE_SEMAPHORE
        # so that the resource audit knows about any cpus we've pinned.
        instance_numa_topology = claim.claimed_numa_topology
        instance.numa_topology = instance_numa_topology
        inst_copy_zero_disk.numa_topology = instance_numa_topology
        # NOTE(melwitt): We don't pass the copy with zero disk here to avoid
        # saving instance.flavor.root_gb=0 to the database.
        self._set_instance_host_and_node(instance, nodename)

        if self.pci_tracker:
            # NOTE(jaypipes): ComputeNode.pci_device_pools is set below
            # in _update_usage_from_instance().
            #  NOTE(melwitt): We don't pass the copy with zero disk here to
            # avoid saving instance.flavor.root_gb=0 to the database.
            self.pci_tracker.claim_instance(context, instance, pci_requests,
                                            instance_numa_topology)

        # WRS: add claim to tracker's list
        # tracked_in_progress is a list of instances that are not yet
        # created but whose resources have been already claimed.
        # The resource audit needs to know of ongoing claims so that
        # disk_available_least is adjusted. Note that this is different
        # from other disk stats like local_gb_used because once the
        #  instance is created, its disk space, along with its qcow
        # backup files, is accounted by the driver (get_disk_available_least())
        # and therefore should no longer be accounted for in _update_usage()
        self.tracked_in_progress.append(instance.uuid)

        # Mark resources in-use and update stats
        self._update_usage_from_instance(context, inst_copy_zero_disk,
                                         nodename, strict=True)

        elevated = context.elevated()
        # persist changes to the compute node:
        self._update(elevated, cn)

        return claim

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def live_migration_claim(self, context, instance, nodename, migration=None,
                             limits=None):
        """Create a claim for a live_migration operation."""
        instance_type = instance.flavor
        image_meta = objects.ImageMeta.from_instance(instance)
        return self._move_claim(context, instance, instance_type, nodename,
                                move_type='live-migration', limits=limits,
                                image_meta=image_meta, migration=migration)

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def rebuild_claim(self, context, instance, nodename, limits=None,
                      image_meta=None, migration=None):
        """Create a claim for a rebuild operation."""
        instance_type = instance.flavor
        # WRS: If boot from volume, image_meta will be empty so get from
        # the instance.
        if not image_meta:
            image_meta = objects.ImageMeta.from_instance(instance)
        return self._move_claim(context, instance, instance_type, nodename,
                                move_type='evacuation', limits=limits,
                                image_meta=image_meta, migration=migration)

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def resize_claim(self, context, instance, instance_type, nodename,
                     image_meta=None, limits=None):
        """Create a claim for a resize or cold-migration move."""
        return self._move_claim(context, instance, instance_type, nodename,
                                image_meta=image_meta, limits=limits)

    def _move_claim(self, context, instance, new_instance_type, nodename,
                    move_type=None, image_meta=None, limits=None,
                    migration=None):
        """Indicate that resources are needed for a move to this host.

        Move can be either a migrate/resize, live-migrate or an
        evacuate/rebuild operation.

        :param context: security context
        :param instance: instance object to reserve resources for
        :param new_instance_type: new instance_type being resized to
        :param nodename: The Ironic nodename selected by the scheduler
        :param image_meta: instance image metadata
        :param move_type: move type - can be one of 'migration', 'resize',
                         'live-migration', 'evacuate'
        :param limits: Dict of oversubscription limits for memory, disk,
        and CPUs
        :param migration: A migration object if one was already created
                          elsewhere for this operation
        :returns: A Claim ticket representing the reserved resources.  This
        should be turned into finalize  a resource claim or free
        resources after the compute operation is finished.
        """
        # TODO(melwitt/jaypipes): Remove this after resource-providers can
        # handle claims and reporting for boot-from-volume.
        inst_copy_zero_disk, new_itype_copy_zero_disk = self._copy_if_bfv(
            instance, new_instance_type)
        image_meta = image_meta or {}
        if migration:
            self._claim_existing_migration(migration, nodename)
        else:
            migration = self._create_migration(context, inst_copy_zero_disk,
                                               new_itype_copy_zero_disk,
                                               nodename, move_type)

        if self.disabled(nodename):
            # compute_driver doesn't support resource tracking, just
            # generate the migration record and continue the resize:
            return claims.NopClaim(migration=migration)

        # get memory overhead required to build this instance:
        overhead = self.driver.estimate_instance_overhead(
            new_itype_copy_zero_disk)
        LOG.debug("Memory overhead for %(flavor)d MB instance; %(overhead)d "
                  "MB", {'flavor': new_itype_copy_zero_disk.memory_mb,
                          'overhead': overhead['memory_mb']})
        LOG.debug("Disk overhead for %(flavor)d GB instance; %(overhead)d "
                  "GB", {'flavor': new_itype_copy_zero_disk.root_gb,
                         'overhead': overhead.get('disk_gb', 0)})
        LOG.debug("CPU overhead for %(flavor)d vCPUs instance; %(overhead)d "
                  "vCPU(s)", {'flavor': new_itype_copy_zero_disk.vcpus,
                              'overhead': overhead.get('vcpus', 0)})

        cn = self.compute_nodes[nodename]

        # TODO(moshele): we are recreating the pci requests even if
        # there was no change on resize. This will cause allocating
        # the old/new pci device in the resize phase. In the future
        # we would like to optimise this.
        # NOTE(melwitt): We don't pass the copy with zero disk here to avoid
        # saving root_gb=0 to the database.
        new_pci_requests = pci_request.get_pci_requests_from_flavor(
            new_instance_type)
        new_pci_requests.instance_uuid = instance.uuid
        # PCI requests come from two sources: instance flavor and
        # SR-IOV ports. SR-IOV ports pci_request don't have an alias_name.
        # On resize merge the SR-IOV ports pci_requests with the new
        # instance flavor pci_requests.
        if instance.pci_requests:
            for request in instance.pci_requests.requests:
                if request.alias_name is None:
                    new_pci_requests.requests.append(request)
        claim = claims.MoveClaim(context, inst_copy_zero_disk, nodename,
                                 new_itype_copy_zero_disk, image_meta, self,
                                 cn, new_pci_requests, overhead=overhead,
                                 limits=limits)

        claim.migration = migration
        claimed_pci_devices_objs = []
        if self.pci_tracker:
            # NOTE(jaypipes): ComputeNode.pci_device_pools is set below
            # in _update_usage_from_instance().
            claimed_pci_devices_objs = self.pci_tracker.claim_instance(
                    context, instance, new_pci_requests,
                    claim.claimed_numa_topology)
        claimed_pci_devices = objects.PciDeviceList(
                objects=claimed_pci_devices_objs)

        # TODO(jaypipes): Move claimed_numa_topology out of the Claim's
        # constructor flow so the Claim constructor only tests whether
        # resources can be claimed, not consume the resources directly.
        mig_context = objects.MigrationContext(
            context=context, instance_uuid=instance.uuid,
            migration_id=migration.id,
            old_numa_topology=instance.numa_topology,
            new_numa_topology=claim.claimed_numa_topology,
            old_pci_devices=instance.pci_devices,
            new_pci_devices=claimed_pci_devices,
            old_pci_requests=instance.pci_requests,
            new_pci_requests=new_pci_requests,
            new_allowed_cpus=hardware.get_vcpu_pin_set())

        def getter(obj, attr, default=None):
            """Method to get object attributes without exception."""
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            else:
                return default

        # WRS: Log migration context creation.
        LOG.info(
            "Migration type:%(ty)s, "
            "source_compute:%(sc)s source_node:%(sn)s, "
            "dest_compute:%(dc)s dest_node:%(dn)s, "
            "new_allowed_cpus=%(allowed)s",
            {'ty': getter(migration, 'migration_type'),
             'sc': getter(migration, 'source_compute'),
             'sn': getter(migration, 'source_node'),
             'dc': getter(migration, 'dest_compute'),
             'dn': getter(migration, 'dest_node'),
             'allowed': getter(mig_context, 'new_allowed_cpus')},
            instance=instance)

        instance.migration_context = mig_context
        inst_copy_zero_disk.migration_context = mig_context
        instance.save()

        # Mark the resources in-use for the resize landing on this
        # compute host:
        self._update_usage_from_migration(context, inst_copy_zero_disk,
                                          migration, nodename, strict=True)
        elevated = context.elevated()
        self._update(elevated, cn)

        return claim

    def _create_migration(self, context, instance, new_instance_type,
                          nodename, move_type=None):
        """Create a migration record for the upcoming resize.  This should
        be done while the COMPUTE_RESOURCES_SEMAPHORE is held so the resource
        claim will not be lost if the audit process starts.
        """
        migration = objects.Migration(context=context.elevated())
        migration.dest_compute = self.host
        migration.dest_node = nodename
        migration.dest_host = self.driver.get_host_ip_addr()
        migration.old_instance_type_id = instance.flavor.id
        migration.new_instance_type_id = new_instance_type.id
        migration.status = 'pre-migrating'
        migration.instance_uuid = instance.uuid
        migration.source_compute = instance.host
        migration.source_node = instance.node
        if move_type:
            migration.migration_type = move_type
        else:
            migration.migration_type = migration_obj.determine_migration_type(
                migration)
        migration.create()
        return migration

    def _claim_existing_migration(self, migration, nodename):
        """Make an existing migration record count for resource tracking.

        If a migration record was created already before the request made
        it to this compute host, only set up the migration so it's included in
        resource tracking. This should be done while the
        COMPUTE_RESOURCES_SEMAPHORE is held.
        """
        migration.dest_compute = self.host
        migration.dest_node = nodename
        migration.dest_host = self.driver.get_host_ip_addr()
        migration.status = 'pre-migrating'
        migration.save()

    def _set_instance_host_and_node(self, instance, nodename):
        """Tag the instance as belonging to this host.  This should be done
        while the COMPUTE_RESOURCES_SEMAPHORE is held so the resource claim
        will not be lost if the audit process starts.
        """
        instance.host = self.host
        instance.launched_on = self.host
        instance.node = nodename
        instance.save()

    def _unset_instance_host_and_node(self, instance):
        """Untag the instance so it no longer belongs to the host.

        This should be done while the COMPUTE_RESOURCES_SEMAPHORE is held so
        the resource claim will not be lost if the audit process starts.
        """
        instance.host = None
        instance.node = None
        instance.save()

    def tracker_dal_update(self, context, instance, going_out=False,
                           live_mig_rollback=False):
        # COMPUTE_RESOURCE_SEMAPHORE should already be taken
        if not instance.node:
            return
        nodename = instance.node
        try:
            cn = self.compute_nodes[nodename]
        except KeyError:
            # Likely we are comming from a host reboot
            return

        if not self.tracked_in_progress:
            migrations = objects.MigrationList.\
                get_in_progress_by_host_and_node(context,
                                                 self.host, nodename)
            if not migrations:
                # we are no longer in cached mode
                cn.disk_available_least = \
                    self.get_disk_available_least(migrations)
                return

        # we are in cached mode
        if live_mig_rollback:
            # we are rolling back from a live-migration. We should leave
            # cached_dal untouched because it was never decremented
            return
        disk_space = instance.get('root_gb', 0) + \
                     instance.get('ephemeral_gb', 0)
        swap = instance.flavor.get('swap', 0)
        disk_space += int(math.ceil((swap * units.Mi) / float(units.Gi)))
        if going_out:
            self.cached_dal += disk_space
        else:
            self.cached_dal -= disk_space
        cn.disk_available_least = self.cached_dal

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def remove_instance_claim(self, context, instance):
        uuid = instance.get('uuid')
        if uuid and uuid in self.tracked_in_progress:
            self.tracked_in_progress.remove(uuid)
            self.tracker_dal_update(context, instance)

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def abort_instance_claim(self, context, instance, nodename):
        """Remove usage from the given instance."""
        # WRS: non-strict pinning accounting when freeing
        self._update_usage_from_instance(context, instance, nodename,
                                         is_removed=True, strict=False)

        uuid = instance.get('uuid')
        if uuid and uuid in self.tracked_in_progress:
            self.tracked_in_progress.remove(uuid)

        instance.clear_numa_topology()
        self._unset_instance_host_and_node(instance)

        self._update(context.elevated(), self.compute_nodes[nodename])

    def _drop_pci_devices(self, instance, nodename, prefix):
        if self.pci_tracker:
            # free old/new allocated pci devices
            pci_devices = self._get_migration_context_resource(
                'pci_devices', instance, prefix=prefix)
            if pci_devices:
                for pci_device in pci_devices:
                    self.pci_tracker.free_device(pci_device, instance)

                dev_pools_obj = self.pci_tracker.stats.to_device_pools_obj()
                self.compute_nodes[nodename].pci_device_pools = dev_pools_obj

    # WRS: Refactor drop_move_claim to cleanup both tracked_migrations and
    # tracked_instances.
    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def drop_move_claim(self, context, instance, nodename,
                        instance_type=None, prefix='new_',
                        live_mig_rollback=False):
        # WRS: get numa_topology based on prefix
        numa_topology = self._get_migration_context_resource('numa_topology',
                                  instance, prefix=prefix)
        # Remove usage for an incoming/outgoing migration on the destination
        # node.
        if instance['uuid'] in self.tracked_migrations:
            migration = self.tracked_migrations.pop(instance['uuid'])

            if not instance_type:
                ctxt = context.elevated()
                instance_type = self._get_instance_type(ctxt, instance, prefix,
                                                        migration)

            if instance_type is not None:
                # TODO(melwitt/jaypipes): Remove this after resource-providers
                # can handle claims and reporting for boot-from-volume.
                itype_copy = self._copy_instance_type_if_bfv(instance,
                                                             instance_type)
                usage = self._get_usage_dict(
                        itype_copy, numa_topology=numa_topology)
                self._drop_pci_devices(instance, nodename, prefix)
                # WRS: non-strict pinning accounting when freeing
                self._update_usage(usage, nodename, sign=-1, strict=False,
                                   from_migration=True)

                ctxt = context.elevated()
                self._update(ctxt, self.compute_nodes[nodename])
        # Remove usage for an instance that is not tracked in migrations (such
        # as on the source node after a migration).
        # NOTE(lbeliveau): On resize on the same node, the instance is
        # included in both tracked_migrations and tracked_instances.
        elif (instance['uuid'] in self.tracked_instances):
            self.tracked_instances.pop(instance['uuid'])
            self._drop_pci_devices(instance, nodename, prefix)
            # TODO(lbeliveau): Validate if numa needs the same treatment.

            if not instance_type:
                instance_type = instance.flavor
            # TODO(melwitt/jaypipes): Remove this after resource-providers
            # can handle claims and reporting for boot-from-volume.
            itype_copy = self._copy_instance_type_if_bfv(instance,
                                                         instance_type)
            usage = self._get_usage_dict(
                        itype_copy, numa_topology=numa_topology)
            # WRS: non-strict pinning accounting when freeing
            self._update_usage(usage, nodename, sign=-1, strict=False,
                               from_migration=True)
            ctxt = context.elevated()
            self._update(ctxt, self.compute_nodes[nodename])

        # NOTE(jaypipes): This sucks, but due to the fact that confirm_resize()
        # only runs on the source host and revert_resize() runs on the
        # destination host, we need to do this here. Basically, what we're
        # doing here is grabbing the existing allocations for this instance
        # from the placement API, dropping the resources in the doubled-up
        # allocation set that refer to the source host UUID and calling PUT
        # /allocations back to the placement API. The allocation that gets
        # PUT'd back to placement will only include the destination host and
        # any shared providers in the case of a confirm_resize operation and
        # the source host and shared providers for a revert_resize operation..
        my_resources = scheduler_utils.resources_from_flavor(instance,
            instance_type or instance.flavor)

        # WRS: need to make sure we're dropping resources based on the correct
        # numa_topology either old or new
        cn = self.compute_nodes[nodename]
        flavor = instance_type or instance.flavor
        num_offline_cpus = scheduler_utils.determine_offline_cpus(flavor,
                                                             numa_topology)
        vcpus = flavor.vcpus - num_offline_cpus
        system_metadata = instance.system_metadata
        image_meta = utils.get_image_from_system_metadata(system_metadata)
        image_props = image_meta.get('properties', {})

        normalized_resources = \
              scheduler_utils.normalized_resources_for_placement_claim(
                  my_resources, cn, vcpus, flavor.extra_specs, image_props,
                  numa_topology)

        operation = 'Confirming'
        source_or_dest = 'source'
        if prefix == 'new_':
            operation = 'Reverting'
            source_or_dest = 'destination'
        LOG.debug("%s resize on %s host. Removing resources claimed on "
                  "provider %s from allocation",
                  operation, source_or_dest, cn.uuid, instance=instance)
        res = self.reportclient.remove_provider_from_instance_allocation(
            instance.uuid, cn.uuid, instance.user_id,
            instance.project_id, normalized_resources)
        if not res:
            LOG.error("Failed to save manipulated allocation when "
                      "%s resize on %s host %s.",
                      operation.lower(), source_or_dest, cn.uuid,
                      instance=instance)

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def update_usage(self, context, instance, nodename):
        """Update the resource usage and stats after a change in an
        instance
        """
        if self.disabled(nodename):
            return

        uuid = instance['uuid']

        # don't update usage for this instance unless it submitted a resource
        # claim first:
        if uuid in self.tracked_instances:
            self._update_usage_from_instance(context, instance, nodename,
                                             strict=False)
            self._update(context.elevated(), self.compute_nodes[nodename])

    def disabled(self, nodename):
        return (nodename not in self.compute_nodes or
                not self.driver.node_is_available(nodename))

    def destroy_instance_and_update_tracker(self, context,
        instance, network_info, block_device_info):

        self.driver.destroy(context, instance, network_info,
                    block_device_info)

        @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
        def _update_dal():
            self.tracker_dal_update(context, instance, going_out=True)
        _update_dal()

    def spawn_instance_and_update_tracker(self, context, instance, image_meta,
                                          injected_files, admin_password,
                                          network_info,
                                          block_device_info):

        self.driver.spawn(context, instance, image_meta,
                                          injected_files, admin_password,
                                          network_info=network_info,
                                          block_device_info=block_device_info)
        self.remove_instance_claim(context, instance)

    def _init_compute_node(self, context, resources):
        """Initialize the compute node if it does not already exist.

        The resource tracker will be inoperable if compute_node
        is not defined. The compute_node will remain undefined if
        we fail to create it or if there is no associated service
        registered.

        If this method has to create a compute node it needs initial
        values - these come from resources.

        :param context: security context
        :param resources: initial values
        """
        nodename = resources['hypervisor_hostname']

        # if there is already a compute node just use resources
        # to initialize
        if nodename in self.compute_nodes:
            cn = self.compute_nodes[nodename]
            self._copy_resources(cn, resources)
            self._setup_pci_tracker(context, cn, resources)
            # WRS: Do not update database here by calling _update().
            # The host numa_topology usage was cleared by _copy_resources(),
            # and we need this recalculated by resource tracker audit.
            # self._update(context, cn)
            return

        # now try to get the compute node record from the
        # database. If we get one we use resources to initialize
        cn = self._get_compute_node(context, nodename)
        if cn:
            self.compute_nodes[nodename] = cn
            self._copy_resources(cn, resources)
            self._setup_pci_tracker(context, cn, resources)
            # WRS: Do not update database here by calling _update().
            # The host numa_topology usage was cleared by _copy_resources(),
            # and we need this recalculated by resource tracker audit.
            # self._update(context, cn)
            return

        # there was no local copy and none in the database
        # so we need to create a new compute node. This needs
        # to be initialized with resource values.
        cn = objects.ComputeNode(context)
        cn.host = self.host
        # WRS: host mapping is already done at service_create
        cn.mapped = 1
        self._copy_resources(cn, resources)
        self.compute_nodes[nodename] = cn
        cn.create()
        LOG.info('Compute node record created for '
                 '%(host)s:%(node)s with uuid: %(uuid)s',
                 {'host': self.host, 'node': nodename, 'uuid': cn.uuid})

        self._setup_pci_tracker(context, cn, resources)
        self._update(context, cn)

    def _setup_pci_tracker(self, context, compute_node, resources):
        if not self.pci_tracker:
            n_id = compute_node.id
            self.pci_tracker = pci_manager.PciDevTracker(context, node_id=n_id)
            if 'pci_passthrough_devices' in resources:
                dev_json = resources.pop('pci_passthrough_devices')
                self.pci_tracker.update_devices_from_hypervisor_resources(
                        dev_json)

            dev_pools_obj = self.pci_tracker.stats.to_device_pools_obj()
            compute_node.pci_device_pools = dev_pools_obj

    def _copy_resources(self, compute_node, resources):
        """Copy resource values to supplied compute_node."""
        # purge old stats and init with anything passed in by the driver
        self.stats.clear()
        self.stats.digest_stats(resources.get('stats'))
        compute_node.stats = copy.deepcopy(self.stats)

        # update the allocation ratios for the related ComputeNode object
        compute_node.ram_allocation_ratio = self.ram_allocation_ratio
        compute_node.cpu_allocation_ratio = self.cpu_allocation_ratio
        compute_node.disk_allocation_ratio = self.disk_allocation_ratio

        # now copy rest to compute_node
        compute_node.update_from_virt_driver(resources)

    def _get_host_metrics(self, context, nodename):
        """Get the metrics from monitors and
        notify information to message bus.
        """
        metrics = objects.MonitorMetricList()
        metrics_info = {}
        for monitor in self.monitors:
            try:
                monitor.populate_metrics(metrics)
            except NotImplementedError:
                LOG.debug("The compute driver doesn't support host "
                          "metrics for  %(mon)s", {'mon': monitor})
            except Exception as exc:
                LOG.warning("Cannot get the metrics from %(mon)s; "
                            "error: %(exc)s",
                            {'mon': monitor, 'exc': exc})
        # TODO(jaypipes): Remove this when compute_node.metrics doesn't need
        # to be populated as a JSONified string.
        metrics = metrics.to_list()
        if len(metrics):
            metrics_info['nodename'] = nodename
            metrics_info['metrics'] = metrics
            metrics_info['host'] = self.host
            metrics_info['host_ip'] = CONF.my_ip
            notifier = rpc.get_notifier(service='compute', host=nodename)
            notifier.info(context, 'compute.metrics.update', metrics_info)
        return metrics

    def update_available_resource(self, context, nodename):
        """Override in-memory calculations of compute node resource usage based
        on data audited from the hypervisor layer.

        Add in resource claims in progress to account for operations that have
        declared a need for resources, but not necessarily retrieved them from
        the hypervisor layer yet.

        :param nodename: Temporary parameter representing the Ironic resource
                         node. This parameter will be removed once Ironic
                         baremetal resource nodes are handled like any other
                         resource in the system.
        """
        LOG.info("Auditing locally available compute resources for "
                  "%(host)s (node: %(node)s)",
                 {'node': nodename,
                  'host': self.host})
        resources = self.driver.get_available_resource(nodename)
        # NOTE(jaypipes): The resources['hypervisor_hostname'] field now
        # contains a non-None value, even for non-Ironic nova-compute hosts. It
        # is this value that will be populated in the compute_nodes table.
        resources['host_ip'] = CONF.my_ip

        # We want the 'cpu_info' to be None from the POV of the
        # virt driver, but the DB requires it to be non-null so
        # just force it to empty string
        if "cpu_info" not in resources or resources["cpu_info"] is None:
            resources["cpu_info"] = ''

        self._verify_resources(resources)

        self._report_hypervisor_resource_view(resources)
        self._update_available_resource(context, resources)

    def _pair_instances_to_migrations(self, migrations, instances):
        instance_by_uuid = {inst.uuid: inst for inst in instances}
        for migration in migrations:
            try:
                migration.instance = instance_by_uuid[migration.instance_uuid]
            except KeyError:
                # NOTE(danms): If this happens, we don't set it here, and
                # let the code either fail or lazy-load the instance later
                # which is what happened before we added this optimization.
                # NOTE(tdurakov) this situation is possible for resize/cold
                # migration when migration is finished but haven't yet
                # confirmed/reverted in that case instance already changed host
                # to destination and no matching happens
                LOG.debug('Migration for instance %(uuid)s refers to '
                              'another host\'s instance!',
                          {'uuid': migration.instance_uuid})

    def get_disk_available_least(self, migrations):
        # if tracked_in_progress is not empty, there is an ongoing instance
        # launch. We avoid reading disk_available_least from the driver here
        # because it could be stale. It is better to use the last good
        # reading and adjust its value in _update_usage()

        # We purge migrations in the "done" status here because that is the
        # state they become after an evacuation, and remain in that state
        # until the source host recovers and re-initializes. If the source
        # host never recovers we would be never coming out of the cached state.
        migrations_in_progress = []
        for migration in migrations:
            if _instance_in_migration_or_resize_state(migration.instance):
                mig_info = {'instance_uuid': migration.instance_uuid}
                mig_info['status'] = migration.status
                migrations_in_progress.append(mig_info)
        if not (self.tracked_in_progress or migrations_in_progress):
            self.cached_dal = self.driver.get_disk_available_least()
        else:
            LOG.info("disk_available_least in cached mode. "
                     "In progress instances:%s migrations:%s",
                     self.tracked_in_progress, migrations_in_progress)
        return self.cached_dal

    @utils.synchronized(COMPUTE_RESOURCE_SEMAPHORE)
    def _update_available_resource(self, context, resources):
        nodename = resources['hypervisor_hostname']
        # Grab all instances assigned to this node:
        instances = objects.InstanceList.get_by_host_and_node(
            context, self.host, nodename,
            expected_attrs=['system_metadata',
                            'numa_topology',
                            'flavor', 'migration_context'])

        # Grab all in-progress migrations:
        # WRS: Move this up as close as possible after the call to
        # get_by_host_and_node() because there is a window where an instance
        # is in the RESIZE_MIGRATED state during the above call but the
        # resize finishes and gets confirmed before this call.
        # If that happens the result is that the instance gets missed by
        # the resource audit.
        migrations = objects.MigrationList.get_in_progress_by_host_and_node(
                context, self.host, nodename)
        self._pair_instances_to_migrations(migrations, instances)

        # WRS: needs to be set before init_compute_node in order to be included
        # in compute_node's _changed_fields
        resources['disk_available_least'] = \
            self.get_disk_available_least(migrations)

        # initialize the compute node object, creating it
        # if it does not already exist.
        self._init_compute_node(context, resources)

        # if we could not init the compute node the tracker will be
        # disabled and we should quit now
        if self.disabled(nodename):
            return

        # Now calculate usage based on instance utilization:
        self._update_usage_from_instances(context, instances, nodename)
        self._update_usage_from_migrations(context, migrations, nodename)

        # Detect and account for orphaned instances that may exist on the
        # hypervisor, but are not in the DB:
        orphans = self._find_orphaned_instances()
        self._update_usage_from_orphans(orphans, nodename)

        cn = self.compute_nodes[nodename]
        # TODO(jgauld): May need to include L3 CAT size for orphans.

        # NOTE(yjiang5): Because pci device tracker status is not cleared in
        # this periodic task, and also because the resource tracker is not
        # notified when instances are deleted, we need remove all usages
        # from deleted instances.
        self.pci_tracker.clean_usage(instances, migrations, orphans)
        dev_pools_obj = self.pci_tracker.stats.to_device_pools_obj()
        cn.pci_device_pools = dev_pools_obj

        metrics = self._get_host_metrics(context, nodename)
        # TODO(pmurray): metrics should not be a json string in ComputeNode,
        # but it is. This should be changed in ComputeNode
        cn.metrics = jsonutils.dumps(metrics)

        # WRS: If the actual free disk space is less than calculated or the
        # actual used disk space is more than calculated, update it so the
        # scheduler doesn't think we have more space than we do.
        local_gb_info = self.driver.get_local_gb_info()
        local_gb_used = local_gb_info.get('used')
        if (local_gb_used is not None and
                    local_gb_used > cn.local_gb_used):
            cn.local_gb_used = local_gb_used
        local_gb_free = local_gb_info.get('free')
        if (local_gb_free is not None and
                    local_gb_free < cn.free_disk_gb):
            cn.free_disk_gb = local_gb_free

        # WRS: Adjust the view of pages used to account for unexpected usages
        # (e.g., orphans, platform overheads) that would make the actual free
        # memory less than what is resource tracked.
        if CONF.adjust_actual_mem_usage:
            self._adjust_actual_mem_usage(context, nodename)

        # update the compute_node
        self._update(context, cn)
        LOG.info('Compute_service record updated for %(host)s:%(node)s',
                  {'host': self.host, 'node': nodename})

        # WRS - move _report_final to reflect _update and metrics
        self._report_final_resource_view(nodename)

    def _add_numa_usage_into_stats(self, compute_node):
        host_numa_topology, jsonify_result = \
            hardware.host_topology_and_format_from_host(compute_node)
        if host_numa_topology:
            vcpus_by_node = {}
            vcpus_used_by_node = {}
            memory_mb_by_node = {}
            memory_mb_used_by_node = {}
            l3_cache_granularity = None
            l3_cache_by_node = {}
            l3_cache_used_by_node = {}
            for cell in host_numa_topology.cells:
                pinned = set(cell.pinned_cpus)
                cpuset = set(cell.cpuset)
                shared = cell.cpu_usage - len(pinned)
                vcpus_by_node[cell.id] = len(cpuset)
                vcpus_used_by_node[cell.id] = \
                    {"shared": shared, "dedicated": len(pinned)}
                mem = {}
                mem_used = {}
                for M in cell.mempages:
                    unit = 'K'
                    size = M.size_kb
                    if M.size_kb >= units.Ki and M.size_kb < units.Mi:
                        unit = 'M'
                        size = M.size_kb / units.Ki
                    if M.size_kb >= units.Mi:
                        unit = 'G'
                        size = M.size_kb / units.Mi
                    mem_unit = "%(sz)s%(U)s" % {'sz': size, 'U': unit}
                    mem[mem_unit] = M.size_kb * M.total / units.Ki
                    mem_used[mem_unit] = M.size_kb * M.used / units.Ki
                memory_mb_by_node[cell.id] = mem
                memory_mb_used_by_node[cell.id] = mem_used
                l3_cache_used = {}
                if cell.l3_size is not None:
                    l3_cache_by_node[cell.id] = cell.l3_size
                    if cell.has_cachetune_cdp:
                        if cell.l3_code_used is not None:
                            l3_cache_used[fields.CacheTuneType.CODE] = \
                                cell.l3_code_used
                        if cell.l3_data_used is not None:
                            l3_cache_used[fields.CacheTuneType.DATA] = \
                                cell.l3_data_used
                    else:
                        if cell.l3_both_used is not None:
                            l3_cache_used[fields.CacheTuneType.BOTH] = \
                                cell.l3_both_used
                l3_cache_used_by_node[cell.id] = l3_cache_used
                l3_cache_granularity = cell.l3_granularity

            self.stats.vcpus_by_node = jsonutils.dumps(vcpus_by_node)
            self.stats.vcpus_used_by_node = jsonutils.dumps(vcpus_used_by_node)
            self.stats.memory_mb_by_node = jsonutils.dumps(memory_mb_by_node)
            self.stats.memory_mb_used_by_node = \
                jsonutils.dumps(memory_mb_used_by_node)
            self.stats.l3_cache_granularity = l3_cache_granularity
            self.stats.l3_cache_by_node = jsonutils.dumps(l3_cache_by_node)
            self.stats.l3_cache_used_by_node = \
                jsonutils.dumps(l3_cache_used_by_node)

    def _get_compute_node(self, context, nodename):
        """Returns compute node for the host and nodename."""
        try:
            return objects.ComputeNode.get_by_host_and_nodename(
                context, self.host, nodename)
        except exception.NotFound:
            LOG.warning("No compute node record for %(host)s:%(node)s",
                        {'host': self.host, 'node': nodename})

    def _report_hypervisor_resource_view(self, resources):
        """Log the hypervisor's view of free resources.

        This is just a snapshot of resource usage recorded by the
        virt driver.

        The following resources are logged:
            - free memory
            - free disk
            - free CPUs
            - assignable PCI devices
        """
        nodename = resources['hypervisor_hostname']
        free_ram_mb = resources['memory_mb'] - resources['memory_mb_used']
        free_disk_gb = resources['local_gb'] - resources['local_gb_used']
        vcpus = resources['vcpus']
        if vcpus:
            free_vcpus = vcpus - resources['vcpus_used']
        else:
            free_vcpus = 'unknown'

        pci_devices = resources.get('pci_passthrough_devices')

        LOG.debug("Hypervisor/Node resource view: "
                  "name=%(node)s "
                  "free_ram=%(free_ram)sMB "
                  "free_disk=%(free_disk)sGB "
                  "free_vcpus=%(free_vcpus)s "
                  "pci_devices=%(pci_devices)s",
                  {'node': nodename,
                   'free_ram': free_ram_mb,
                   'free_disk': free_disk_gb,
                   'free_vcpus': free_vcpus,
                   'pci_devices': pci_devices})

    def _report_final_resource_view(self, nodename):
        """Report final calculate of physical memory, used virtual memory,
        disk, usable vCPUs, used virtual CPUs and PCI devices,
        including instance calculations and in-progress resource claims. These
        values will be exposed via the compute node table to the scheduler.
        """
        cn = self.compute_nodes[nodename]
        vcpus = cn.vcpus
        if vcpus:
            tcpu = vcpus
            ucpu = cn.vcpus_used
            LOG.debug("Total usable vcpus: %(tcpu)s, "
                      "total allocated vcpus: %(ucpu)s",
                      {'tcpu': vcpus,
                       'ucpu': ucpu})
        else:
            tcpu = 0
            ucpu = 0
            LOG.info(_("Free VCPU information unavailable"))
        pci_stats = (list(cn.pci_device_pools) if
            cn.pci_device_pools else [])
        LOG.info("Final resource view: "
                 "name=%(node)s "
                 "phys_ram=%(phys_ram)sMB "
                 "used_ram=%(used_ram)sMB "
                 "phys_disk=%(phys_disk)sGB "
                 "used_disk=%(used_disk)sGB "
                 "free_disk=%(free_disk)sGB "
                 "total_vcpus=%(total_vcpus)s "
                 "used_vcpus=%(used_vcpus)s "
                 "pci_stats=%(pci_stats)s",
                 {'node': nodename,
                  'phys_ram': cn.memory_mb,
                  'used_ram': cn.memory_mb_used,
                  'phys_disk': cn.local_gb,
                  'used_disk': cn.local_gb_used,
                  'free_disk': cn.free_disk_gb,
                  'total_vcpus': tcpu,
                  'used_vcpus': ucpu,
                  'pci_stats': pci_stats})

        # WRS - Ignore printing extended resources
        if not utils.is_libvirt_compute(cn):
            return

        # WRS - display per-numa resources
        host_numa_topology, jsonify_result = \
            hardware.host_topology_and_format_from_host(cn)
        if host_numa_topology is None:
            host_numa_topology = objects.NUMATopology(cells=[])

        # WRS - display per-numa memory usage
        for cell in host_numa_topology.cells:
            LOG.info(
                'Numa node=%(node)d; '
                'memory: %(T)5d MiB total, %(A)5d MiB avail',
                {'node': cell.id, 'T': cell.memory, 'A': cell.avail_memory})
        for cell in host_numa_topology.cells:
            mem = []
            for M in cell.mempages:
                unit = 'K'
                size = M.size_kb
                if M.size_kb >= units.Ki and M.size_kb < units.Mi:
                    unit = 'M'
                    size = M.size_kb / units.Ki
                if M.size_kb >= units.Mi:
                    unit = 'G'
                    size = M.size_kb / units.Mi
                m = '%(sz)s%(U)s: %(T).0f MiB total, %(A).0f MiB avail' % \
                    {'sz': size, 'U': unit,
                     'T': M.size_kb * M.total / units.Ki,
                     'A': M.size_kb * (M.total - M.used) / units.Ki,
                     }
                mem.append(m)
            LOG.info('Numa node=%(node)d; per-pgsize: %(pgsize)s',
                     {'node': cell.id, 'pgsize': '; '.join(mem)})

        # WRS - display per-numa cpu usage
        for cell in host_numa_topology.cells:
            cpuset = set(cell.cpuset)
            pinned = set(cell.pinned_cpus)
            unpinned = cpuset - pinned
            shared = cell.cpu_usage - len(pinned)
            LOG.info(
                'Numa node=%(node)d; cpu_usage:%(usage).3f, pcpus:%(pcpus)d, '
                'pinned:%(P)d, shared:%(S).3f, unpinned:%(U)d; '
                'pinned_cpulist:%(LP)s, '
                'unpinned_cpulist:%(LU)s',
                {'usage': cell.cpu_usage,
                 'node': cell.id, 'pcpus': len(cpuset),
                 'P': len(pinned), 'S': shared, 'U': len(unpinned),
                 'LP': utils.list_to_range(sorted(list(pinned))) or '-',
                 'LU': utils.list_to_range(sorted(list(unpinned))) or '-',
                 })

        # L3 CAT Support
        if ((cn.l3_closids is not None) and
                (cn.l3_closids_used is not None) and
                host_numa_topology.cells and
                host_numa_topology.cells[0].has_cachetune):
            LOG.info(_('L3 CAT: closids: %(total)d total, %(avail)d avail'),
                      {'total': cn.l3_closids,
                       'avail': (cn.l3_closids -
                                 cn.l3_closids_used),
                       })
            for cell in host_numa_topology.cells:
                if cell.has_cachetune_cdp:
                    used = '{code} KiB code used, {data} KiB data used'.\
                        format(code=cell.l3_code_used, data=cell.l3_data_used)
                else:
                    used = '{both} KiB both used'.\
                        format(both=cell.l3_both_used)
                LOG.info(_(
                    'L3 CAT: Numa node=%(node)d; '
                    '%(size)d KiB total, '
                    '%(avail)d KiB avail, '
                    '%(gran)d KiB gran; '
                    '%(used)s'),
                    {'node': cell.id,
                     'size': cell.l3_size,
                     'avail': cell.avail_cache,
                     'gran': cell.l3_granularity,
                     'used': used,
                     })
        else:
            LOG.info(_("L3 CAT unavailable"))

    def _resource_change(self, compute_node):
        """Check to see if any resources have changed."""
        nodename = compute_node.hypervisor_hostname
        old_compute = self.old_resources[nodename]
        if not obj_base.obj_equal_prims(
                compute_node, old_compute, ['updated_at']):
            self.old_resources[nodename] = copy.deepcopy(compute_node)
            return True
        return False

    def _update(self, context, compute_node):
        """Update partial stats locally and populate them to Scheduler."""
        self._add_numa_usage_into_stats(compute_node)
        compute_node.stats = copy.deepcopy(self.stats)
        if not self._resource_change(compute_node):
            return
        nodename = compute_node.hypervisor_hostname
        compute_node.save()
        # Persist the stats to the Scheduler
        try:
            inv_data = self.driver.get_inventory(nodename)
            _normalize_inventory_from_cn_obj(inv_data, compute_node)
            self.scheduler_client.set_inventory_for_provider(
                compute_node.uuid,
                compute_node.hypervisor_hostname,
                inv_data,
            )
        except NotImplementedError:
            # Eventually all virt drivers will return an inventory dict in the
            # format that the placement API expects and we'll be able to remove
            # this code branch
            self.scheduler_client.update_compute_node(compute_node)

        if self.pci_tracker:
            self.pci_tracker.save(context)

    # WRS:extension - normalized vCPU accounting
    def normalized_vcpus(self, usage, nodename):
        """Return normalized vCPUs

        Determine the fractional number of vCPUs used for an instance
        based on normalized accounting.
        """
        cn = self.compute_nodes[nodename]
        vcpus = float(usage.get('vcpus', 0))
        if vcpus == 0:
            return 0

        # 'usage' can be instance or flavor, want to get at
        # extra_specs as efficiently as possible.
        if 'extra_specs' in usage:
            extra_specs = usage['extra_specs']
        else:
            if 'flavor' in usage:
                extra_specs = usage['flavor'].get('extra_specs', {})
            elif 'instance_type_id' in usage:
                flavor = objects.Flavor.get_by_id(
                    nova.context.get_admin_context(read_deleted='yes'),
                    usage['instance_type_id'])
                extra_specs = flavor.extra_specs
            else:
                extra_specs = {}

        # Get instance numa topology
        instance_numa_topology = None
        if 'numa_topology' in usage:
            instance_numa_topology = usage.get('numa_topology')

        if 'hw:cpu_policy' not in extra_specs \
            and instance_numa_topology \
            and instance_numa_topology.cells[0].cpu_policy:
            extra_specs['hw:cpu_policy'] = \
                instance_numa_topology.cells[0].cpu_policy

        # WRS: When called from _update_usage_from_migration(), the 'usage'
        # arg comes from a flavor, so it doesn't have system_metadata.
        # However in that case numa_topology will have the correct cpu_policy
        # and cpu_thread_policy so add them to image properties.
        image_props = {}
        if 'system_metadata' in usage:
            system_metadata = usage['system_metadata']
            image_meta = utils.get_image_from_system_metadata(system_metadata)
            image_props = image_meta.get('properties', {})
        elif instance_numa_topology:
            image_props['hw_cpu_policy'] = \
                instance_numa_topology.cells[0].cpu_policy
            image_props['hw_cpu_thread_policy'] = \
                instance_numa_topology.cells[0].cpu_thread_policy

        # Get host numa topology
        host_numa_topology, _fmt = hardware.host_topology_and_format_from_host(
            cn)

        # Get set of reserved thread sibling pcpus that cannot be allocated
        # when using 'isolate' cpu_thread_policy.
        reserved = hardware.get_reserved_thread_sibling_pcpus(
                instance_numa_topology, host_numa_topology)
        threads_per_core = hardware._get_threads_per_core(host_numa_topology)

        # Normalize vcpus accounting
        vcpus = hardware.normalized_vcpus(vcpus=vcpus,
                                          reserved=reserved,
                                          extra_specs=extra_specs,
                                          image_props=image_props,
                                          ratio=CONF.cpu_allocation_ratio,
                                          threads_per_core=threads_per_core)
        return vcpus

    # WRS: add update_affinity
    def _update_usage(self, usage, nodename, sign=1, update_affinity=True,
                      strict=True, from_migration=False):
        # WRS: tracker debug logging
        if CONF.compute_resource_debug:
            # Get parent calling functions
            p5_fname = sys._getframe(5).f_code.co_name
            p4_fname = sys._getframe(4).f_code.co_name
            p3_fname = sys._getframe(3).f_code.co_name
            p2_fname = sys._getframe(2).f_code.co_name
            p1_fname = sys._getframe(1).f_code.co_name
            LOG.info(
                '_update_usage: '
                'caller=%(p5)s / %(p4)s / %(p3)s / %(p2)s / %(p1)s, '
                'sign=%(sign)r, USAGE: host:%(host)s uuid:%(uuid)s '
                'name:%(name)s display_name:%(display_name)s '
                'min_vcpus:%(min_vcpus)s vcpus:%(vcpus)s '
                'max_vcpus:%(max_vcpus)s '
                'memory_mb:%(memory_mb)s root_gb:%(root_gb)s '
                'ephemeral_gb:%(ephemeral_gb)s '
                'instance_type_id:%(instance_type_id)s '
                'old_flavor:%(old_flavor)s new_flavor:%(new_flavor)s '
                'vm_mode:%(vm_mode)s task_state:%(task_state)s '
                'vm_state:%(vm_state)s power_state:%(power_state)s '
                'launched_at:%(launched_at)s terminated_at:%(terminated_at)s '
                '%(numa_topology)r',
                {'p5': p5_fname, 'p4': p4_fname, 'p3': p3_fname,
                 'p2': p2_fname, 'p1': p1_fname,
                 'sign': sign,
                 'host': usage.get('host'),
                 'uuid': usage.get('uuid'),
                 'name': usage.get('name'),
                 'display_name': usage.get('display_name'),
                 'min_vcpus': usage.get('min_vcpus'),
                 'vcpus': usage.get('vcpus'),
                 'max_vcpus': usage.get('max_vcpus'),
                 'memory_mb': usage.get('memory_mb'),
                 'root_gb': usage.get('root_gb'),
                 'ephemeral_gb': usage.get('ephemeral_gb'),
                 'instance_type_id': usage.get('instance_type_id'),
                 'old_flavor': usage.get('old_flavor'),
                 'new_flavor': usage.get('new_flavor'),
                 'vm_mode': usage.get('vm_mode'),
                 'task_state': usage.get('task_state'),
                 'vm_state': usage.get('vm_state'),
                 'power_state': usage.get('power_state'),
                 'launched_at': usage.get('launched_at'),
                 'terminated_at': usage.get('terminated_at'),
                 'numa_topology': usage.get('numa_topology'),
                 })
            self._populate_usage(nodename, id=0)

        mem_usage = usage['memory_mb']
        disk_usage = usage.get('root_gb', 0)
        vcpus_usage = self.normalized_vcpus(usage, nodename)

        overhead = self.driver.estimate_instance_overhead(usage)
        mem_usage += overhead['memory_mb']
        disk_usage += overhead.get('disk_gb', 0)
        vcpus_usage += overhead.get('vcpus', 0)

        cn = self.compute_nodes[nodename]
        cn.memory_mb_used += sign * mem_usage
        cn.local_gb_used += sign * disk_usage
        cn.local_gb_used += sign * usage.get('ephemeral_gb', 0)
        cn.vcpus_used += sign * vcpus_usage

        # L3 CAT Support
        numa_topology = usage.get('numa_topology')
        if ((numa_topology is not None) and
                any(cell.cachetune_requested
                    for cell in numa_topology.cells)):
            cn.l3_closids_used += sign * 1

        if usage.get('uuid') in self.tracked_in_progress or from_migration:
            cn.disk_available_least -= \
                sign * usage.get('root_gb', 0)
            cn.disk_available_least -= \
                sign * usage.get('ephemeral_gb', 0)
            swap = usage.get('swap', 0)
            cn.disk_available_least -= \
                sign * int(math.ceil((swap * units.Mi) / float(units.Gi)))

        # free ram and disk may be negative, depending on policy:
        cn.free_ram_mb = cn.memory_mb - cn.memory_mb_used
        cn.free_disk_gb = cn.local_gb - cn.local_gb_used

        cn.running_vms = self.stats.num_instances

        # Calculate the numa usage
        free = sign == -1
        updated_numa_topology = hardware.get_host_numa_usage_from_instance(
                cn, usage, free, strict=strict)
        cn.numa_topology = updated_numa_topology

        # WRS: update the affinity of instances with non-dedicated CPUs.
        if update_affinity:
            hardware.update_floating_affinity(cn)

        # WRS: tracker debug logging
        if CONF.compute_resource_debug:
            self._populate_usage(nodename, id=1)
            self._display_usage()

    # WRS: tracker debug logging
    def _populate_usage(self, nodename, id=0):
        """populate before and after usage display lines"""
        global _usage_debug
        _usage_debug[id] = list()

        reference = str(id)
        if id == 0:
            reference = 'BEFORE: '
        elif id == 1:
            reference = 'AFTER:  '

        cn = self.compute_nodes[nodename]

        line = (
            '%(ref)s'
            'workload: %(workload)s, vms: %(vms)s, '
            'vcpus: %(vcpu_used).3f used, %(vcpu_tot).3f total' %
            {'ref': reference,
             'workload': cn.current_workload,
             'vms': cn.running_vms,
             'vcpu_tot': cn.vcpus,
             'vcpu_used': cn.vcpus_used,
             })
        _usage_debug[id].append(line)

        line = (
            '%(ref)s'
            'memory: %(mem_used)6d used, %(mem_tot)6d total (MiB); '
            'disk: %(disk_used)3d used, %(disk_tot)3d total, '
            '%(disk_free)3d free, %(disk_least)3d least_avail (GiB)' %
            {'ref': reference,
             'mem_tot': cn.memory_mb,
             'mem_used': cn.memory_mb_used,
             'disk_tot': cn.local_gb,
             'disk_used': cn.local_gb_used,
             'disk_free': cn.free_disk_gb,
             'disk_least': cn.disk_available_least or 0,
             })
        _usage_debug[id].append(line)
        line = (
            '%(ref)s'
            'L3-closids: %(closids_used)2d used, %(closids_tot)2d total' %
            {'ref': reference,
             'closids_tot': cn.l3_closids,
             'closids_used': cn.l3_closids_used,
             })
        _usage_debug[id].append(line)

        host_numa_topology, jsonify_result = \
            hardware.host_topology_and_format_from_host(cn)
        if host_numa_topology is None:
            host_numa_topology = objects.NUMATopology(cells=[])
        for cell in host_numa_topology.cells:
            line = (
                '%(ref)sNuma node=%(node)d; '
                'memory: %(T)5d MiB total, %(A)5d MiB avail' %
                {'ref': reference,
                 'node': cell.id,
                 'T': cell.memory, 'A': cell.avail_memory})
            _usage_debug[id].append(line)

        for cell in host_numa_topology.cells:
            mem = []
            for M in cell.mempages:
                unit = 'K'
                size = M.size_kb
                if M.size_kb >= units.Ki and M.size_kb < units.Mi:
                    unit = 'M'
                    size = M.size_kb / units.Ki
                if M.size_kb >= units.Mi:
                    unit = 'G'
                    size = M.size_kb / units.Mi
                m = '%(sz)s%(U)s: %(T)5.0f MiB total, %(A)5.0f MiB avail' % \
                    {'sz': size, 'U': unit,
                     'T': M.size_kb * M.total / units.Ki,
                     'A': M.size_kb * (M.total - M.used) / units.Ki}
                mem.append(m)
            line = (
                '%(ref)sNuma node=%(node)d; per-pgsize: %(pgsize)s' %
                {'ref': reference,
                 'node': cell.id,
                 'pgsize': '; '.join(mem)})
            _usage_debug[id].append(line)

        for cell in host_numa_topology.cells:
            cpuset = set(cell.cpuset)
            pinned = set(cell.pinned_cpus)
            unpinned = cpuset - pinned
            shared = cell.cpu_usage - len(pinned)
            line = (
                '%(ref)sNuma node=%(node)d; cpu_usage:%(usage)6.3f, '
                'pcpus:%(pcpus)2d, '
                'pinned:%(P)2d, shared:%(S)6.3f, unpinned:%(U)2d; map:%(M)s; '
                'pinned_cpulist:%(LP)s, '
                'unpinned_cpulist:%(LU)s' %
                {'ref': reference,
                 'usage': cell.cpu_usage,
                 'node': cell.id,
                 'pcpus': len(cpuset),
                 'P': len(pinned), 'S': shared,
                 'U': len(unpinned),
                 'M': ''.join(
                     'P' if s in pinned else 'U' for s in cpuset) or '-',
                 'LP': utils.list_to_range(sorted(list(pinned))) or '-',
                 'LU': utils.list_to_range(sorted(list(unpinned))) or '-',
                 })
            _usage_debug[id].append(line)

        for cell in host_numa_topology.cells:
            line = (
                '%(ref)sNuma node=%(node)d; L3-cache-usage: '
                '%(B)6d both KiB, %(C)6d code KiB, %(D)6d data KiB' %
                {'ref': reference,
                 'usage': cell.cpu_usage,
                 'node': cell.id,
                 'B': cell.l3_both_used,
                 'C': cell.l3_code_used,
                 'D': cell.l3_data_used,
                 })
            _usage_debug[id].append(line)

    # WRS: tracker debug logging
    def _display_usage(self):
        """display before and after usage, one line at a time"""
        global _usage_debug
        if not (_usage_debug[0] and _usage_debug[1]):
            return
        for (before, after) in zip(_usage_debug[0], _usage_debug[1]):
            LOG.info(before)
            LOG.info(after)

    def _get_migration_context_resource(self, resource, instance,
                                        prefix='new_'):
        migration_context = instance.migration_context
        resource = prefix + resource
        if migration_context and resource in migration_context:
            return getattr(migration_context, resource)
        return None

    def _update_usage_from_migration(self, context, instance, migration,
                                     nodename, strict=True):
        """Update usage for a single migration.  The record may
        represent an incoming or outbound migration.
        """
        def getter(obj, attr, default=None):
            """Method to get object attributes without exception."""
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            else:
                return default

        uuid = migration.instance_uuid

        incoming = (migration.dest_compute == self.host and
                    migration.dest_node == nodename)
        outbound = (migration.source_compute == self.host and
                    migration.source_node == nodename)
        same_node = (incoming and outbound)

        record = self.tracked_instances.get(uuid, None)
        itype = None
        numa_topology = None
        sign = 0
        same_node_old = False
        if same_node:
            # Same node resize. Record usage for the 'new_' resources.  This
            # is executed on resize_claim().
            if (instance['instance_type_id'] ==
                    migration.old_instance_type_id):
                itype = self._get_instance_type(context, instance, 'new_',
                        migration)
                numa_topology = self._get_migration_context_resource(
                    'numa_topology', instance)
                # Allocate pci device(s) for the instance.
                sign = 1
            else:
                # The instance is already set to the new flavor (this is done
                # by the compute manager on finish_resize()), hold space for a
                # possible revert to the 'old_' resources.
                # NOTE(lbeliveau): When the periodic audit timer gets
                # triggered, the compute usage gets reset.  The usage for an
                # instance that is migrated to the new flavor but not yet
                # confirmed/reverted will first get accounted for by
                # _update_usage_from_instances().  This method will then be
                # called, and we need to account for the '_old' resources
                # (just in case).
                itype = self._get_instance_type(context, instance, 'old_',
                        migration)
                numa_topology = self._get_migration_context_resource(
                    'numa_topology', instance, prefix='old_')
                same_node_old = True

        elif incoming and not record:
            # instance has not yet migrated here:
            itype = self._get_instance_type(context, instance, 'new_',
                    migration)
            numa_topology = self._get_migration_context_resource(
                'numa_topology', instance)
            # Allocate pci device(s) for the instance.
            sign = 1

        elif outbound and not record:
            # instance migrated, but record usage for a possible revert:
            itype = self._get_instance_type(context, instance, 'old_',
                    migration)
            numa_topology = self._get_migration_context_resource(
                'numa_topology', instance, prefix='old_')

        if itype:
            cn = self.compute_nodes[nodename]
            # TODO(melwitt/jaypipes): Remove this after resource-providers
            # can handle claims and reporting for boot-from-volume.
            itype_copy = self._copy_instance_type_if_bfv(instance, itype)

            usage = self._get_usage_dict(
                        itype_copy, numa_topology=numa_topology)
            if self.pci_tracker and sign:
                self.pci_tracker.update_pci_for_instance(
                    context, instance, sign=sign)
            self._update_usage(usage, nodename, strict=strict,
                               from_migration=(incoming and not same_node_old))
            if self.pci_tracker:
                obj = self.pci_tracker.stats.to_device_pools_obj()
                cn.pci_device_pools = obj
            else:
                obj = objects.PciDevicePoolList()
                cn.pci_device_pools = obj
            self.tracked_migrations[uuid] = migration

            # WRS: Display migrations that audit includes via _update_usage.
            change = None
            if incoming:
                change = 'incoming'
            if outbound:
                change = 'outbound'
            if same_node:
                change = 'same_node'
            topo = utils.format_instance_numa_topology(
                numa_topology=numa_topology, instance=instance, delim=', ')
            # Get pci_devices associated with this 'host' and 'nodename'.
            pci_devices = pci_manager.get_instance_pci_devs_by_host_and_node(
                instance, request_id='all', host=self.host, nodename=nodename)
            if pci_devices:
                devs = '; pci_devices='
                devs += pci_utils.format_instance_pci_devices(
                    pci_devices=pci_devices, delim='; ')
            else:
                devs = ''
            LOG.info(
                'Migration(id=%(mid)s, change=%(change)s, '
                'status=%(status)s, type=%(type)s); '
                'sign=%(sign)d, id=%(name)s, name=%(display_name)s, '
                'source_compute=%(sc)s, source_node=%(sn)s, '
                'dest_compute=%(dc)s, dest_node=%(dn)s, '
                'numa_topology=%(topo)s, %(pci_devs)s',
                {'mid': migration.id, 'change': change,
                 'status': migration.status,
                 'type': migration.migration_type,
                 'sign': 1,
                 'name': getter(instance, 'name'),
                 'display_name': getter(instance, 'display_name'),
                 'sc': migration.source_compute,
                 'sn': migration.source_node,
                 'dc': migration.dest_compute,
                 'dn': migration.dest_node,
                 'topo': topo,
                 'pci_devs': devs,
                 }, instance=instance)

    def _update_usage_from_migrations(self, context, migrations, nodename):
        filtered = {}
        instances = {}
        self.tracked_migrations.clear()

        # do some defensive filtering against bad migrations records in the
        # database:
        for migration in migrations:
            uuid = migration.instance_uuid

            try:
                if uuid not in instances:
                    instances[uuid] = migration.instance
            except exception.InstanceNotFound as e:
                # migration referencing deleted instance
                LOG.debug('Migration instance not found: %s', e)
                continue

            # skip migration if instance isn't in a migration or resize state:
            if not _instance_in_migration_or_resize_state(instances[uuid]):
                id = migration.id if hasattr(migration, 'id') else None
                type = (migration.migration_type
                        if hasattr(migration, 'migration_type') else None)
                status = (migration.status
                          if hasattr(migration, 'status') else None)
                LOG.warning("Instance not migrating, skipping migration: "
                            "id=%(id)s, type=%(type)s, status=%(status)s",
                            {'id': id,
                            'type': type,
                            'status': status},
                            instance_uuid=uuid)
                continue

            # filter to most recently updated migration for each instance:
            other_migration = filtered.get(uuid, None)
            # NOTE(claudiub): In Python 3, you cannot compare NoneTypes.
            if other_migration:
                om = other_migration
                other_time = om.updated_at or om.created_at
                migration_time = migration.updated_at or migration.created_at
                if migration_time > other_time:
                    filtered[uuid] = migration
            else:
                filtered[uuid] = migration

        for migration in filtered.values():
            instance = instances[migration.instance_uuid]
            # Skip migration if it doesn't match the instance migration id.
            # This can happen if we have a stale migration record.
            # We want to proceed if instance.migration_context is None
            # (see test_update_available_resources_migration_no_context).
            if (instance.migration_context is not None and
                    instance.migration_context.migration_id != migration.id):
                LOG.warning("Instance migration %(im)s doesn't match "
                            "migration %(m)s, skipping migration.",
                     {'im': instance.migration_context.migration_id,
                     'm': migration.id})
                continue

            try:
                # WRS: non-strict pinning accounting for resource audit
                self._update_usage_from_migration(context, instance, migration,
                                                  nodename, strict=False)
            except exception.FlavorNotFound:
                LOG.warning("Flavor could not be found, skipping migration.",
                            instance_uuid=instance.uuid)
                continue

    # WRS: add update_affinity
    def _update_usage_from_instance(self, context, instance, nodename,
            is_removed=False, require_allocation_refresh=False,
            update_affinity=True, strict=True):
        """Update usage for a single instance."""
        def getter(obj, attr, default=None):
            """Method to get object attributes without exception."""
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            else:
                return default

        uuid = instance['uuid']
        is_new_instance = uuid not in self.tracked_instances
        # NOTE(sfinucan): Both brand new instances as well as instances that
        # are being unshelved will have is_new_instance == True
        is_removed_instance = not is_new_instance and (is_removed or
            instance['vm_state'] in vm_states.ALLOW_RESOURCE_REMOVAL)

        # Avoid changing the original instance in the case of boot from volume
        inst_copy_zero_disk = None

        if is_new_instance:
            # TODO(melwitt/jaypipes): Remove this after resource-providers
            # can handle claims and reporting for boot-from-volume.
            inst_copy_zero_disk, _ = self._copy_if_bfv(instance)

            self.tracked_instances[uuid] = obj_base.obj_to_primitive(
                    inst_copy_zero_disk)
            sign = 1

        if is_removed_instance:
            # TODO(melwitt/jaypipes): Remove this after resource-providers
            # can handle claims and reporting for boot-from-volume.
            inst_copy_zero_disk, _ = self._copy_if_bfv(instance)

            self.tracked_instances.pop(uuid)
            sign = -1

        cn = self.compute_nodes[nodename]
        self.stats.update_stats_for_instance(inst_copy_zero_disk or instance,
                                             is_removed_instance)

        # if it's a new or deleted instance:
        if is_new_instance or is_removed_instance:
            if self.pci_tracker:
                # NOTE(melwitt): We don't pass the copy with zero disk here to
                # avoid saving instance.flavor.root_gb=0 to the database.
                self.pci_tracker.update_pci_for_instance(context,
                                                         instance,
                                                         sign=sign)
            # WRS: This covers rare cases where scheduler allocation is
            # incorrect and needs to be auto-corrected on the next resource
            # audit. However if the instance is migrating, leave the
            # allocations alone.
            if not _instance_in_migration_or_resize_state(instance):
                self.scheduler_client.reportclient.update_instance_allocation(
                    cn, instance, sign)
            # new instance, update compute node resource usage:
            self._update_usage(self._get_usage_dict(inst_copy_zero_disk),
                               nodename, sign=sign,
                               update_affinity=update_affinity, strict=strict)

            # WRS: Display instances that audit includes via _update_usage.
            pstate = instance.get('power_state')
            if pstate is None:
                pstate = power_state.NOSTATE
            topo = utils.format_instance_numa_topology(
                numa_topology=instance.get('numa_topology'),
                instance=instance, delim='; ')
            # Get pci_devices associated with this instance
            pci_devices = pci_manager.get_instance_pci_devs_by_host_and_node(
                instance, request_id='all', host=self.host, nodename=nodename)
            if pci_devices:
                devs = '; pci_devices='
                devs += pci_utils.format_instance_pci_devices(
                    pci_devices=pci_devices, delim='; ')
            else:
                devs = ''
            LOG.info(
                'sign=%(sign)s, id=%(name)s, name=%(display_name)s, '
                'vm_mode=%(vm)s, task_state=%(task)s, power_state=%(power)s, '
                'numa_topology=%(topo)s, %(pci_devs)s',
                {'sign': sign,
                'name': getter(instance, 'name'),
                'display_name': getter(instance, 'display_name'),
                'vm': instance.get('vm_state'),
                'task': instance.get('task_state'),
                'power': power_state.STATE_MAP[pstate],
                'topo': topo,
                'pci_devs': devs,
                }, instance=instance)

        cn.current_workload = self.stats.calculate_workload()

        if self.pci_tracker:
            obj = self.pci_tracker.stats.to_device_pools_obj()
            cn.pci_device_pools = obj
        else:
            cn.pci_device_pools = objects.PciDevicePoolList()

    def _update_usage_from_instances(self, context, instances, nodename):
        """Calculate resource usage based on instance utilization.  This is
        different than the hypervisor's view as it will account for all
        instances assigned to the local compute host, even if they are not
        currently powered on.
        """
        self.tracked_instances.clear()

        cn = self.compute_nodes[nodename]
        # set some initial values, reserve room for host/hypervisor:
        cn.local_gb_used = CONF.reserved_host_disk_mb / 1024
        cn.memory_mb_used = CONF.reserved_host_memory_mb
        cn.vcpus_used = CONF.reserved_host_cpus
        cn.free_ram_mb = (cn.memory_mb - cn.memory_mb_used)
        cn.free_disk_gb = (cn.local_gb - cn.local_gb_used)
        cn.current_workload = 0
        cn.running_vms = 0

        # NOTE(jaypipes): In Pike, we need to be tolerant of Ocata compute
        # nodes that overwrite placement allocations to look like what the
        # resource tracker *thinks* is correct. When an instance is
        # migrated from an Ocata compute node to a Pike compute node, the
        # Pike scheduler will have created a "doubled-up" allocation that
        # contains allocated resources against both the source and
        # destination hosts. The Ocata source compute host, during its
        # update_available_resource() periodic call will find the instance
        # in its list of known instances and will call
        # update_instance_allocation() in the report client. That call will
        # pull the allocations for the instance UUID which will contain
        # both the source and destination host providers in the allocation
        # set. Seeing that this is different from what the Ocata source
        # host thinks it should be and will overwrite the allocation to
        # only be an allocation against itself.
        #
        # And therefore, here we need to have Pike compute hosts
        # "correct" the improper healing that the Ocata source host did
        # during its periodic interval. When the instance is fully migrated
        # to the Pike compute host, the Ocata compute host will find an
        # allocation that refers to itself for an instance it no longer
        # controls and will *delete* all allocations that refer to that
        # instance UUID, assuming that the instance has been deleted. We
        # need the destination Pike compute host to recreate that
        # allocation to refer to its own resource provider UUID.
        #
        # For Pike compute nodes that migrate to either a Pike compute host
        # or a Queens compute host, we do NOT want the Pike compute host to
        # be "healing" allocation information. Instead, we rely on the Pike
        # scheduler to properly create allocations during scheduling.
        compute_version = objects.Service.get_minimum_version(
            context, 'nova-compute')
        has_ocata_computes = compute_version < 22

        # Some drivers (ironic) still need the allocations to be
        # fixed up, as they transition the way their inventory is reported.
        require_allocation_refresh = (
            has_ocata_computes or
            self.driver.requires_allocation_refresh)

        # L3 CAT Support - include default CLOS
        has_cachetune = self.driver._has_cachetune_support()
        if has_cachetune:
            cn.l3_closids_used = 1
        else:
            cn.l3_closids_used = 0

        for instance in instances:
            if instance.task_state == task_states.RESIZE_MIGRATED:
                # WRS: We need to ignore these instances because
                # instance.nova_topology still points at the old value.
                # _update_usage_from_migrations() will handle them.
                LOG.debug("Instance %s task state is RESIZE_MIGRATED, "
                          "ignoring in _update_usage_from_instances().",
                          instance.uuid)
                continue
            # WRS: During post live migration, destination host will update
            # instance host and numa_topology but it seems that these are
            # separate db operations.  If this is happening just as the
            # resource audit on the source host is getting the list of
            # instances from the db, the instance numa_topology can be updated
            # to the destination but the host is not.  So skip those cases here
            # based on MIGRATING state and available migration context and let
            # the migration part of the audit update the resource accounting.
            if instance.task_state == task_states.MIGRATING and \
                   instance.migration_context is not None:
                LOG.debug("Instance %s task state is MIGRATING, "
                          "ignoring in _update_usage_from_instances().",
                          instance.uuid)
                continue
            if instance.vm_state not in vm_states.ALLOW_RESOURCE_REMOVAL:
                self._update_usage_from_instance(context, instance, nodename,
                    require_allocation_refresh=require_allocation_refresh,
                    strict=False)

        self._remove_deleted_instances_allocations(context, cn)
        # WRS: update the affinity of instances with non-dedicated CPUs.
        hardware.update_floating_affinity(cn)

    def _remove_deleted_instances_allocations(self, context, cn):
        # NOTE(jaypipes): All of this code sucks. It's basically dealing with
        # all the corner cases in move, local delete, unshelve and rebuild
        # operations for when allocations should be deleted when things didn't
        # happen according to the normal flow of events where the scheduler
        # always creates allocations for an instance
        known_instances = set(self.tracked_instances.keys())
        allocations = self.reportclient.get_allocations_for_resource_provider(
                cn.uuid) or {}
        read_deleted_context = context.elevated(read_deleted='yes')
        for instance_uuid, alloc in allocations.items():
            if instance_uuid in known_instances:
                LOG.debug("Instance %s actively managed on this compute host "
                          "and has allocations in placement: %s.",
                          instance_uuid, alloc)
                continue
            try:
                instance = objects.Instance.get_by_uuid(read_deleted_context,
                                                        instance_uuid,
                                                        expected_attrs=[])
            except exception.InstanceNotFound:
                # The instance isn't even in the database. Either the scheduler
                # _just_ created an allocation for it and we're racing with the
                # creation in the cell database, or the instance was deleted
                # and fully archived before we got a chance to run this. The
                # former is far more likely than the latter. Avoid deleting
                # allocations for a building instance here.
                LOG.info("Instance %(uuid)s has allocations against this "
                         "compute host but is not found in the database.",
                         {'uuid': instance_uuid},
                         exc_info=False)
                continue

            if instance.deleted:
                # The instance is gone, so we definitely want to remove
                # allocations associated with it.
                # NOTE(jaypipes): This will not be true if/when we support
                # cross-cell migrations...
                LOG.debug("Instance %s has been deleted (perhaps locally). "
                          "Deleting allocations that remained for this "
                          "instance against this compute host: %s.",
                          instance_uuid, alloc)
                self.reportclient.delete_allocation_for_instance(instance_uuid)
                continue
            if not instance.host:
                # Allocations related to instances being scheduled should not
                # be deleted if we already wrote the allocation previously.
                LOG.debug("Instance %s has been scheduled to this compute "
                          "host, the scheduler has made an allocation "
                          "against this compute node but the instance has "
                          "yet to start. Skipping heal of allocation: %s.",
                          instance_uuid, alloc)
                continue
            if (instance.host == cn.host and
                    instance.node == cn.hypervisor_hostname):
                # The instance is supposed to be on this compute host but is
                # not in the list of actively managed instances.
                LOG.warning("Instance %s is not being actively managed by "
                            "this compute host but has allocations "
                            "referencing this compute host: %s. Skipping "
                            "heal of allocation because we do not know "
                            "what to do.", instance_uuid, alloc)
                continue
            if instance.host != cn.host:
                # The instance has been moved to another host either via a
                # migration, evacuation or unshelve in between the time when we
                # ran InstanceList.get_by_host_and_node(), added those
                # instances to RT.tracked_instances and the above
                # Instance.get_by_uuid() call. We SHOULD attempt to remove any
                # allocations that reference this compute host if the VM is in
                # a stable terminal state (i.e. it isn't in a state of waiting
                # for resize to confirm/revert), however if the destination
                # host is an Ocata compute host, it will delete the allocation
                # that contains this source compute host information anyway and
                # recreate an allocation that only refers to itself. So we
                # don't need to do anything in that case. Just log the
                # situation here for debugging information but don't attempt to
                # delete or change the allocation.
                LOG.debug("Instance %s has been moved to another host %s(%s). "
                          "There are allocations remaining against the source "
                          "host that might need to be removed: %s.",
                          instance_uuid, instance.host, instance.node, alloc)
                # WRS: remove allocations if instance is not in migrating state
                if not _instance_in_migration_or_resize_state(instance):
                    self.reportclient.remove_provider_from_instance_allocation(
                             instance.uuid, cn.uuid, instance.user_id,
                             instance.project_id, alloc['resources'])
                continue

    def delete_allocation_for_evacuated_instance(self, context, instance, node,
                                                 node_type='source'):
        self._delete_allocation_for_moved_instance(context,
            instance, node, 'evacuated', node_type)

    def delete_allocation_for_migrated_instance(self, context, instance, node):
        self._delete_allocation_for_moved_instance(context, instance, node,
                                                   'migrated')

    def _delete_allocation_for_moved_instance(
            self, context, instance, node, move_type, node_type='source'):
        # Clean up the instance allocation from this node in placement
        my_resources = scheduler_utils.resources_from_flavor(
            instance, instance.flavor)

        if node not in self.compute_nodes:
            # WRS: during evacuation, this is called before
            # _init_compute_node() so compute_nodes may not be initialized
            # TODO(GK) with fix in fb968e18 this may not be required anymore
            self.compute_nodes[node] = \
                self._get_compute_node(context, node)

        # WRS: Use instance fields as instance was not resized
        cn = self.compute_nodes[node]
        system_metadata = instance.system_metadata
        image_meta = utils.get_image_from_system_metadata(system_metadata)
        image_props = image_meta.get('properties', {})
        normalized_resources = \
              scheduler_utils.normalized_resources_for_placement_claim(
                  my_resources, cn, instance.vcpus,
                  instance.flavor.extra_specs, image_props,
                  instance.numa_topology)

        res = self.reportclient.remove_provider_from_instance_allocation(
            instance.uuid, cn.uuid, instance.user_id,
            instance.project_id, normalized_resources)
        if not res:
            LOG.error("Failed to clean allocation of %s "
                      "instance on the %s node %s",
                      move_type, node_type, cn.uuid, instance=instance)

    def delete_allocation_for_failed_resize(self, instance, node, flavor,
                                            image):
        """Delete instance allocations for the node during a failed resize

        :param instance: The instance being resized/migrated.
        :param node: The node provider on which the instance should have
            allocations to remove. If this is a resize to the same host, then
            the new_flavor resources are subtracted from the single allocation.
        :param flavor: This is the new_flavor during a resize.
        """
        resources = scheduler_utils.resources_from_flavor(instance, flavor)
        cn = self.compute_nodes[node]

        # WRS: as resize failed prior to finishing undo the resource claim
        # based on the new flavor to reverse what was done in the scheduler
        offline_cpus = scheduler_utils.determine_offline_cpus(flavor,
                                                     instance.numa_topology)
        vcpus = flavor.vcpus - offline_cpus
        image_meta_obj = objects.ImageMeta.from_dict(image)
        numa_topology = hardware.numa_get_constraints(flavor, image_meta_obj)
        normalized_resources = \
              scheduler_utils.normalized_resources_for_placement_claim(
                  resources, cn, vcpus, flavor.extra_specs,
                  image['properties'], numa_topology)

        res = self.reportclient.remove_provider_from_instance_allocation(
            instance.uuid, cn.uuid, instance.user_id, instance.project_id,
            normalized_resources)
        if not res:
            if instance.instance_type_id == flavor.id:
                operation = 'migration'
            else:
                operation = 'resize'
            LOG.error('Failed to clean allocation after a failed '
                      '%(operation)s on node %(node)s',
                      {'operation': operation, 'node': cn.uuid},
                      instance=instance)

    def _find_orphaned_instances(self):
        """Given the set of instances and migrations already account for
        by resource tracker, sanity check the hypervisor to determine
        if there are any "orphaned" instances left hanging around.

        Orphans could be consuming memory and should be accounted for in
        usage calculations to guard against potential out of memory
        errors.
        """
        uuids1 = frozenset(self.tracked_instances.keys())
        uuids2 = frozenset(self.tracked_migrations.keys())
        uuids = uuids1 | uuids2

        usage = self.driver.get_per_instance_usage()
        vuuids = frozenset(usage.keys())

        orphan_uuids = vuuids - uuids
        orphans = [usage[uuid] for uuid in orphan_uuids]
        return orphans

    def _update_usage_from_orphans(self, orphans, nodename):
        """Include orphaned instances in usage."""
        for orphan in orphans:
            memory_mb = orphan['memory_mb']

            LOG.warning("Detected running orphan instance: %(uuid)s "
                        "(consuming %(memory_mb)s MB memory)",
                        {'uuid': orphan['uuid'], 'memory_mb': memory_mb})
            if CONF.adjust_actual_mem_usage:
                continue

            # just record memory usage for the orphan
            usage = {'memory_mb': memory_mb}
            self._update_usage(usage, nodename, strict=False)

    def delete_allocation_for_shelve_offloaded_instance(self, instance):
        self.reportclient.delete_allocation_for_instance(instance.uuid)

    def _verify_resources(self, resources):
        resource_keys = ["vcpus", "memory_mb", "local_gb", "cpu_info",
                         "vcpus_used", "memory_mb_used", "local_gb_used",
                         "numa_topology"]

        missing_keys = [k for k in resource_keys if k not in resources]
        if missing_keys:
            reason = _("Missing keys: %s") % missing_keys
            raise exception.InvalidInput(reason=reason)

    def _get_instance_type(self, context, instance, prefix, migration):
        """Get the instance type from instance."""
        stashed_flavors = migration.migration_type in ('resize',)
        if stashed_flavors:
            return getattr(instance, '%sflavor' % prefix)
        else:
            # NOTE(ndipanov): Certain migration types (all but resize)
            # do not change flavors so there is no need to stash
            # them. In that case - just get the instance flavor.
            return instance.flavor

    def _get_usage_dict(self, object_or_dict, **updates):
        """Make a usage dict _update methods expect.

        Accepts a dict or an Instance or Flavor object, and a set of updates.
        Converts the object to a dict and applies the updates.

        param object_or_dict: instance or flavor as an object or just a dict
        :param updates: key-value pairs to update the passed object.
                        Currently only considers 'numa_topology', all other
                        keys are ignored.

        :returns: a dict with all the information from object_or_dict updated
                  with updates
        """
        usage = {}
        if isinstance(object_or_dict, objects.Instance):
            usage = {'memory_mb': object_or_dict.flavor.memory_mb,
                     # WRS: get vcpus from instance object directly so that
                     # it reflects actual used vcpus after scaling
                     'vcpus': object_or_dict.vcpus,
                     'root_gb': object_or_dict.flavor.root_gb,
                     'ephemeral_gb': object_or_dict.flavor.ephemeral_gb,
                     'numa_topology': object_or_dict.numa_topology,
                     'uuid': object_or_dict.uuid}
            # WRS: need to add in flavor and system_metadata so we
            # can properly normalize vcpu usage
            usage['flavor'] = object_or_dict.flavor
            if 'system_metadata' in object_or_dict:
                usage['system_metadata'] = object_or_dict.system_metadata

            for field in ['host', 'name', 'display_name', 'min_vcpus',
                          'max_vcpus', 'instance_type_id', 'old_flavor',
                          'new_flavor', 'vm_mode', 'task_state', 'vm_state',
                          'power_state', 'launched_at', 'terminated_at']:
                if hasattr(object_or_dict, field):
                    usage[field] = getattr(object_or_dict, field)
                else:
                    usage[field] = None
        elif isinstance(object_or_dict, objects.Flavor):
            usage = obj_base.obj_to_primitive(object_or_dict)
        else:
            usage.update(object_or_dict)

        for key in ('numa_topology',):
            if key in updates:
                usage[key] = updates[key]
        return usage

    def _adjust_actual_mem_usage(self, context, nodename):
        """Adjust host numa topology view of pages 'used' when the actual free
        pages available is less than what is resource tracked. This accounts
        for orphans, excessive qemu overheads, and runaway platform resources.
        """
        cn = self.compute_nodes[nodename]

        host_numa_topology, jsonify_result = \
            hardware.host_topology_and_format_from_host(cn)
        # NOTE(jgauld): We don't expect to return here on real hardware.
        if host_numa_topology is None:
            return
        if not jsonify_result:
            return

        # Get dictionary of actual free memory pages per numa-cell, per-pgsize
        actual_free_mempages = self.driver.get_actual_free_mempages()
        # NOTE(jgauld): We don't expect to return here on real hardware.
        if not actual_free_mempages:
            return

        update_numa_topology = False
        for cell in host_numa_topology.cells:
            mem = []
            for mempage in cell.mempages:
                try:
                    actual_free = \
                        actual_free_mempages[cell.id][mempage.size_kb]
                except KeyError:
                    actual_free = None
                    LOG.error('Could not get actual_free_mempages: '
                              'cellid=%(node)s, pgsize=%(size)s',
                              {'node': cell.id,
                               'size': mempage.size_kb})
                if actual_free is not None:
                    used_old, used_adjusted = mempage.adjust_used(actual_free)
                    if used_old != used_adjusted:
                        update_numa_topology = True
                        adjust_MiB = mempage.size_kb * (used_adjusted -
                                                        used_old) / units.Ki
                        cn.memory_mb_used += adjust_MiB
                        cn.free_ram_mb -= adjust_MiB

                        unit = 'K'
                        size = mempage.size_kb
                        if ((mempage.size_kb >= units.Ki) and
                            (mempage.size_kb < units.Mi)):
                            unit = 'M'
                            size = mempage.size_kb / units.Ki
                        if mempage.size_kb >= units.Mi:
                            unit = 'G'
                            size = mempage.size_kb / units.Mi
                        m = '%(sz)s%(U)s: %(old).0f MiB used, ' \
                            '%(adj).0f MiB used adj, %(A).0f MiB avail adj' % \
                            {'sz': size, 'U': unit,
                             'old': mempage.size_kb * used_old / units.Ki,
                             'adj': mempage.size_kb * mempage.used / units.Ki,
                             'A': mempage.size_kb *
                                  (mempage.total - mempage.used) / units.Ki,
                            }
                        mem.append(m)
            if mem:
                LOG.warning(
                    'Numa node=%(node)d; Adjusted memory: '
                    'per-pgsize: %(pgsize)s',
                    {'node': cell.id, 'pgsize': '; '.join(mem)})

        if update_numa_topology:
            cn.numa_topology = host_numa_topology._to_json()
