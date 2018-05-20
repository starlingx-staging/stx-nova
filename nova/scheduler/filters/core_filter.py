# Copyright (c) 2011 OpenStack Foundation
# Copyright (c) 2012 Justin Santa Barbara
#
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
# Copyright (c) 2013-2017 Wind River Systems, Inc.
#

from oslo_log import log as logging

from nova.i18n import _LW
from nova.scheduler import filters
from nova.scheduler.filters import utils
from nova.virt import hardware

LOG = logging.getLogger(__name__)


class BaseCoreFilter(filters.BaseHostFilter):

    RUN_ON_REBUILD = False

    def _get_cpu_allocation_ratio(self, host_state, spec_obj):
        raise NotImplementedError

    def host_passes(self, host_state, spec_obj):
        """Return True if host has sufficient CPU cores.

        :param host_state: nova.scheduler.host_manager.HostState
        :param spec_obj: filter options
        :return: boolean
        """
        if not host_state.vcpus_total:
            # Fail safe
            LOG.warning(_LW("VCPUs not set; assuming CPU collection broken"))
            return True

        instance_vcpus = spec_obj.vcpus
        cpu_allocation_ratio = self._get_cpu_allocation_ratio(host_state,
                                                              spec_obj)
        vcpus_total = host_state.vcpus_total

        # WRS: this will be needed further down
        extra_specs = spec_obj.flavor.extra_specs
        image_props = spec_obj.image.properties

        # Only provide a VCPU limit to compute if the virt driver is reporting
        # an accurate count of installed VCPUs. (XenServer driver does not)
        if vcpus_total > 0:
            host_state.limits['vcpu'] = vcpus_total

            # Do not allow an instance to overcommit against itself, only
            # against other instances.
            unshared_vcpus = hardware.unshared_vcpus(instance_vcpus,
                                                     extra_specs)
            if unshared_vcpus > host_state.vcpus_total:
                LOG.debug("%(host_state)s does not have %(instance_vcpus)d "
                      "unshared cpus before overcommit, it only has %(cpus)d",
                      {'host_state': host_state,
                       'instance_vcpus': unshared_vcpus,
                       'cpus': host_state.vcpus_total})
                msg = ('Insufficient total vcpus: req:%(req)s, '
                       'avail:%(cpus)s' % {'req': instance_vcpus,
                       'cpus': host_state.vcpus_total})
                self.filter_reject(host_state, spec_obj, msg)
                return False

        free_vcpus = vcpus_total - host_state.vcpus_used
        # WRS:extension - normalized vCPU accounting.  host_state.vcpus_used
        # is now reported in floating-point.
        host_numa_topology, _fmt = hardware.host_topology_and_format_from_host(
            host_state)
        threads_per_core = hardware._get_threads_per_core(host_numa_topology)
        normalized_instance_vcpus = hardware.normalized_vcpus(
            vcpus=instance_vcpus,
            reserved=set(),
            extra_specs=extra_specs,
            image_props=image_props,
            ratio=cpu_allocation_ratio,
            threads_per_core=threads_per_core)
        if free_vcpus < normalized_instance_vcpus:
            LOG.debug("%(host_state)s does not have %(instance_vcpus)f "
                      "usable vcpus, it only has %(free_vcpus)f usable "
                      "vcpus",
                      {'host_state': host_state,
                       'instance_vcpus': normalized_instance_vcpus,
                       'free_vcpus': free_vcpus})
            msg = ('Insufficient vcpus: req:%(req)s, avail:%(avail)s' %
                   {'req': instance_vcpus, 'avail': free_vcpus})
            self.filter_reject(host_state, spec_obj, msg)
            return False

        return True


class CoreFilter(BaseCoreFilter):
    """CoreFilter filters based on CPU core utilization."""

    def _get_cpu_allocation_ratio(self, host_state, spec_obj):
        return host_state.cpu_allocation_ratio


class AggregateCoreFilter(BaseCoreFilter):
    """AggregateCoreFilter with per-aggregate CPU subscription flag.

    Fall back to global cpu_allocation_ratio if no per-aggregate setting found.
    """

    def _get_cpu_allocation_ratio(self, host_state, spec_obj):
        aggregate_vals = utils.aggregate_values_from_key(
            host_state,
            'cpu_allocation_ratio')
        try:
            ratio = utils.validate_num_values(
                aggregate_vals, host_state.cpu_allocation_ratio, cast_to=float)
        except ValueError as e:
            LOG.warning(_LW("Could not decode cpu_allocation_ratio: '%s'"), e)
            ratio = host_state.cpu_allocation_ratio

        return ratio
