# Copyright (c) 2013 Intel, Inc.
# Copyright (c) 2013 OpenStack Foundation
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

import copy

from oslo_config import cfg
from oslo_log import log as logging
import six

from nova import exception
from nova.objects import fields
from nova.objects import pci_device_pool
from nova.pci import utils
from nova.pci import whitelist


CONF = cfg.CONF
LOG = logging.getLogger(__name__)


class PciDeviceStats(object):

    """PCI devices summary information.

    According to the PCI SR-IOV spec, a PCI physical function can have up to
    256 PCI virtual functions, thus the number of assignable PCI functions in
    a cloud can be big. The scheduler needs to know all device availability
    information in order to determine which compute hosts can support a PCI
    request. Passing individual virtual device information to the scheduler
    does not scale, so we provide summary information.

    Usually the virtual functions provided by a host PCI device have the same
    value for most properties, like vendor_id, product_id and class type.
    The PCI stats class summarizes this information for the scheduler.

    The pci stats information is maintained exclusively by compute node
    resource tracker and updated to database. The scheduler fetches the
    information and selects the compute node accordingly. If a compute
    node is selected, the resource tracker allocates the devices to the
    instance and updates the pci stats information.

    This summary information will be helpful for cloud management also.
    """

    pool_keys = ['product_id', 'vendor_id', 'numa_node', 'dev_type']

    def __init__(self, stats=None, dev_filter=None):
        super(PciDeviceStats, self).__init__()
        # NOTE(sbauza): Stats are a PCIDevicePoolList object
        self.pools = [pci_pool.to_dict()
                      for pci_pool in stats] if stats else []
        # self.pools.sort(key=lambda item: len(item))
        self.pools.sort(key=self.get_key)
        self.dev_filter = dev_filter or whitelist.Whitelist(
            CONF.pci.passthrough_whitelist)

    @staticmethod
    def _pools_prettyprint(pools):
        _pools = "\n"
        for pool in pools:
            # Devices are not exported to the scheduler
            devices = []
            if 'devices' in pool.keys():
                devices = [str(device.address) for device
                                               in pool.get('devices')]
            else:
                devices = pool.get('count')

            fmt = ('[vendor:{}, product:{}, numa:{}, physnet:{}, ' +
                   'class:{}, dev_type:{}]: {}\n')
            _pools += fmt.format(pool.get('vendor_id'),
                                 pool.get('product_id'),
                                 pool.get('numa_node'),
                                 pool.get('physical_network'),
                                 pool.get('class_id'),
                                 pool.get('dev_type'),
                                 devices)
        return _pools

    def _equal_properties(self, dev, entry, matching_keys):
        return all(dev.get(prop) == entry.get(prop)
                   for prop in matching_keys)

    def _find_pool(self, dev_pool):
        """Return the first pool that matches dev."""
        for pool in self.pools:
            pool_keys = pool.copy()
            del pool_keys['count']
            del pool_keys['devices']

            # WRS: 'configured' is not a pool key.
            del pool_keys['configured']

            if (len(pool_keys.keys()) == len(dev_pool.keys()) and
                self._equal_properties(dev_pool, pool_keys, dev_pool.keys())):
                return pool

    def _create_pool_keys_from_dev(self, dev):
        """create a stats pool dict that this dev is supposed to be part of

        Note that this pool dict contains the stats pool's keys and their
        values. 'count' and 'devices' are not included.
        """
        # Don't add a device that doesn't have a matching device spec.
        # This can happen during initial sync up with the controller
        devspec = self.dev_filter.get_devspec(dev)
        if not devspec:
            return
        tags = devspec.get_tags()
        pool = {k: getattr(dev, k) for k in self.pool_keys}
        if tags:
            pool.update(tags)
        return pool

    def add_device(self, dev, sync=False, do_append=True):
        """Add a device to its matching pool.

        :param dev: A PCI device to add to one of the pools.  The device is
            then available for allocation to an instance.
        :param sync: If this flag is set to True it specifies that this device
            is being synchronized with the set of discovered devices by the
            hypervisor.  This is only done at nova compute startup.  This is
            how we know how many PCI devices have been configured (versus
            'count' that specifies how many are available).  If this flag is
            set to False it specifies that the PCI device should be put
            back to the pool after it became available again (for e.g. after
            an instance is terminate).
        :param do_append: If this flag is set to True it specifies that the
            PCI device should be added to the pools.  If this flag is set to
            Flase it specifies that this device should not be added to the
            pool.  This scenario happen when the nova compute process is
            started and initialized and the list of PCI is synchronized with
            the hypervisor.  In this particular case some instances are
            already running and those devices should only be accounted as
            configured and not be added to the list of available devices.
        """
        dev_pool = self._create_pool_keys_from_dev(dev)
        if dev_pool:
            pool = self._find_pool(dev_pool)
            if not pool:
                dev_pool['count'] = 0
                dev_pool['configured'] = 0
                dev_pool['devices'] = []
                self.pools.append(dev_pool)
                # self.pools.sort(key=lambda item: len(item))
                self.pools.sort(key=self.get_key)
                pool = dev_pool

            # WRS: Do not add allocated PCI devices to the pool.  These are
            # reported by the hypervisor on already running instances.  Only
            # available devices should be added to the pool.
            if do_append:
                pool['count'] += 1
                pool['devices'].append(dev)

            # WRS: On nova compute process boot up, PciDevTracker initialize
            # the pool with the PCI devices discovered from the hypervisor.
            # PciDevTracker during init is the only one calling this function
            # with 'sync=True'.  PciDevice objects are created in the database
            # in set_hvdevs.
            if sync:
                pool['configured'] += 1

    @staticmethod
    def _decrease_pool_count(pool_list, pool, count=1):
        """Decrement pool's size by count.

        If pool becomes empty, remove pool from pool_list.
        """
        if pool['count'] > count:
            pool['count'] -= count
            count = 0
        else:
            count -= pool['count']
            pool_list.remove(pool)
        return count

    def remove_device(self, dev):
        """Remove one device from the first pool that it matches."""
        LOG.info("Removing device %s", dev.address)
        dev_pool = self._create_pool_keys_from_dev(dev)
        if dev_pool:
            pool = self._find_pool(dev_pool)
            if not pool:
                raise exception.PciDevicePoolEmpty(
                    compute_node_id=dev.compute_node_id, address=dev.address)
            pool['devices'].remove(dev)
            self._decrease_pool_count(self.pools, pool)
        LOG.info("Pool is now: %s", self._pools_prettyprint(self.pools))

    def get_free_devs(self):
        free_devs = []
        for pool in self.pools:
            free_devs.extend(pool['devices'])
        return free_devs

    # WRS: Originally (upstream) _consume_requests is consume_requests.  This
    # was broken down in two pieces to more easily implement best-effort.
    # Also, for tracking down allocation of PCI devices per PCI request, this
    # function was reduced to handling only one PCI request at a time (for
    # loop over PCI requests is now done in consume_requests).
    def _consume_requests(self, request, numa_cells=None,
                          log_error=True):
        numa = None
        if numa_cells:
            numa = [n.id for n in numa_cells]
        LOG.info("Consuming PCI requests on numa %s", numa)
        alloc_devices = []
        count = request.count
        spec = request.spec
        # For now, keep the same algorithm as during scheduling:
        # a spec may be able to match multiple pools.
        pools = self._filter_pools_for_spec(self.pools, spec)
        if numa_cells:
            pools = self._filter_pools_for_numa_cells(pools, numa_cells)
        pools = self._filter_non_requested_pfs(request, pools)
        # Failed to allocate the required number of devices
        # Return the devices already allocated back to their pools
        if sum([pool['count'] for pool in pools]) < count:
            # WRS: This routine called multiple times. Only log errors
            # last time through.
            if log_error:
                LOG.error("Failed to allocate PCI devices for instance."
                          " Unassigning devices back to pools."
                          " This should not happen, since the scheduler"
                          " should have accurate information, and"
                          " allocation during claims is controlled via a"
                          " hold on the compute node semaphore")
            for d in range(len(alloc_devices)):
                self.add_device(alloc_devices.pop())
            return None

        for pool in pools:
            if pool['count'] >= count:
                num_alloc = count
            else:
                num_alloc = pool['count']
            count -= num_alloc
            pool['count'] -= num_alloc
            for d in range(num_alloc):
                pci_dev = pool['devices'].pop()
                self._handle_device_dependents(pci_dev)
                pci_dev.request_id = request.request_id
                alloc_devices.append(pci_dev)
            if count == 0:
                break
        LOG.info("Allocated devices %(devs)s, pool is now: %(p)s",
                 {'devs': [str(dev.address) for dev in alloc_devices],
                  'p': self._pools_prettyprint(self.pools)})

        return alloc_devices

    # WRS: Add back PCI devices on the pools.
    def _cleanup(self, alloc_devices):
        for d in range(len(alloc_devices)):
            self.add_device(alloc_devices.pop())

    # WRS: Redesigned to keep track of devices so they could be properly
    # cleaned up.  If an individual PCI requests fails, then cleanup the
    # accumulated PCI devices.  In theory this should not happen since
    # the scheduler should have valid view on the pools status (but lets not
    # take any chances ...).
    def consume_requests(self, pci_requests, numa_cells=None, pci_strict=True):
        alloc_devices = []
        for r in pci_requests:
            # WRS: PCI request strict match on same NUMA node.
            _alloc_devices = self._consume_requests(r,
                                                    numa_cells,
                                                    log_error=pci_strict)
            if not _alloc_devices:
                if pci_strict:
                    self._cleanup(alloc_devices)
                    raise exception.PciDeviceRequestFailed(
                        requests=pci_requests)
                else:
                    # WRS: PCI request matching any NUMA node.
                    _alloc_devices = self._consume_requests(r, None)
                    if not _alloc_devices:
                        self._cleanup(alloc_devices)
                        raise exception.PciDeviceRequestFailed(
                            requests=pci_requests)

            # WRS: Accumulate all the PCI devices from individual PCI
            # requests so that if there is an error we can put them back
            # in the pools.
            alloc_devices.extend(_alloc_devices)

        return alloc_devices

    def _handle_device_dependents(self, pci_dev):
        """Remove device dependents or a parent from pools.

        In case the device is a PF, all of it's dependent VFs should
        be removed from pools count, if these are present.
        When the device is a VF, it's parent PF pool count should be
        decreased, unless it is no longer in a pool.
        """
        if pci_dev.dev_type == fields.PciDeviceType.SRIOV_PF:
            vfs_list = pci_dev.child_devices
            if vfs_list:
                for vf in vfs_list:
                    self.remove_device(vf)
        elif pci_dev.dev_type == fields.PciDeviceType.SRIOV_VF:
            try:
                parent = pci_dev.parent_device
                # Make sure not to decrease PF pool count if this parent has
                # been already removed from pools
                if parent in self.get_free_devs():
                    self.remove_device(parent)
            except exception.PciDeviceNotFound:
                return

    @staticmethod
    def _filter_pools_for_spec(pools, request_specs):
        return [pool for pool in pools
                if utils.pci_device_prop_match(pool, request_specs)]

    @staticmethod
    def _filter_pools_for_numa_cells(pools, numa_cells):
        # Some systems don't report numa node info for pci devices, in
        # that case None is reported in pci_device.numa_node, by adding None
        # to numa_cells we allow assigning those devices to instances with
        # numa topology
        numa_cells = [None] + [cell.id for cell in numa_cells]
        # filter out pools which numa_node is not included in numa_cells
        return [pool for pool in pools if any(utils.pci_device_prop_match(
                                pool, [{'numa_node': cell}])
                                              for cell in numa_cells)]

    def _filter_non_requested_pfs(self, request, matching_pools):
        # Remove SRIOV_PFs from pools, unless it has been explicitly requested
        # This is especially needed in cases where PFs and VFs has the same
        # product_id.
        if all(spec.get('dev_type') != fields.PciDeviceType.SRIOV_PF for
               spec in request.spec):
            matching_pools = self._filter_pools_for_pfs(matching_pools)
        return matching_pools

    @staticmethod
    def _filter_pools_for_pfs(pools):
        return [pool for pool in pools
                if not pool.get('dev_type') == fields.PciDeviceType.SRIOV_PF]

    def _apply_request(self, pools, request, numa_cells=None):
        # NOTE(vladikr): This code maybe open to race conditions.
        # Two concurrent requests may succeed when called support_requests
        # because this method does not remove related devices from the pools
        LOG.info("request: %s", request)
        numa = None
        if numa_cells:
            numa = [n.id for n in numa_cells]
        LOG.info("Applying PCI request on numa %(numa)s: %(request)s",
                 {'numa': numa, 'request': request})

        count = request.count
        matching_pools = self._filter_pools_for_spec(pools, request.spec)
        LOG.info("matching_pools: %s",
                 self._pools_prettyprint(matching_pools))
        if numa_cells:
            matching_pools = self._filter_pools_for_numa_cells(matching_pools,
                                                          numa_cells)
            LOG.info("matching_pools with numa_cells: %s",
                     self._pools_prettyprint(matching_pools))
        matching_pools = self._filter_non_requested_pfs(request,
                                                        matching_pools)
        if sum([pool['count'] for pool in matching_pools]) < count:
            return False
        else:
            for pool in matching_pools:
                count = self._decrease_pool_count(pools, pool, count)
                if not count:
                    break

        return True

    def support_requests(self, requests, numa_cells=None, pci_strict=True):
        """Check if the pci requests can be met.

        Scheduler checks compute node's PCI stats to decide if an
        instance can be scheduled into the node. Support does not
        mean real allocation.
        If numa_cells is provided then only devices contained in
        those nodes are considered.
        """
        # note (yjiang5): this function has high possibility to fail,
        # so no exception should be triggered for performance reason.
        pools = copy.deepcopy(self.pools)
        for r in requests:
            # WRS: PCI request strict match on same NUMA node.
            if not self._apply_request(pools, r, numa_cells):
                # WRS: PCI request matching any NUMA node.
                if pci_strict or not self._apply_request(pools, r, None):
                    return False

        return True

    def apply_requests(self, requests, numa_cells=None, pci_strict=True):
        """Apply PCI requests to the PCI stats.

        This is used in multiple instance creation, when the scheduler has to
        maintain how the resources are consumed by the instances.
        If numa_cells is provided then only devices contained in
        those nodes are considered.
        """
        for r in requests:
            # WRS: PCI request strict match on same NUMA node.
            if not self._apply_request(self.pools, r, numa_cells):
                # WRS: PCI request matching any NUMA node.
                if pci_strict or not self._apply_request(self.pools, r, None):
                    raise exception.PciDeviceRequestFailed(requests=requests)

    @staticmethod
    def get_key(pool):
        # WRS: Make sort comparator more deterministic using specific keys.
        # depending on the device type.
        return (pool['product_id'], pool['vendor_id'], pool['numa_node'],
            pool.get('dev_type'), pool.get('physical_network'),
            pool.get('class_id'))

    def __iter__(self):
        # 'devices' shouldn't be part of stats
        pools = []
        for pool in self.pools:
            tmp = {k: v for k, v in pool.items() if k != 'devices'}
            pools.append(tmp)
        return iter(pools)

    def clear(self):
        """Clear all the stats maintained."""
        self.pools = []

    def __eq__(self, other):
        return self.pools == other.pools

    if six.PY2:
        def __ne__(self, other):
            return not (self == other)

    def to_device_pools_obj(self):
        """Return the contents of the pools as a PciDevicePoolList object."""
        stats = [x for x in self]
        return pci_device_pool.from_pci_stats(stats)
