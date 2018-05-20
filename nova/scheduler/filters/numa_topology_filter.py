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

from nova import objects
from nova.objects import fields
from nova.scheduler import filters
from nova import utils
from nova.virt import hardware

LOG = logging.getLogger(__name__)


class NUMATopologyFilter(filters.BaseHostFilter):
    """Filter on requested NUMA topology."""

    RUN_ON_REBUILD = True

    def _satisfies_cpu_policy(self, host_state, extra_specs, image_props,
                              details):
        """Check that the host_state provided satisfies any available
        CPU policy requirements.
        """
        host_topology, _ = hardware.host_topology_and_format_from_host(
            host_state)
        # NOTE(stephenfin): There can be conflicts between the policy
        # specified by the image and that specified by the instance, but this
        # is not the place to resolve these. We do this during scheduling.
        cpu_policy = [extra_specs.get('hw:cpu_policy'),
                      image_props.get('hw_cpu_policy')]
        cpu_thread_policy = [extra_specs.get('hw:cpu_thread_policy'),
                             image_props.get('hw_cpu_thread_policy')]

        if not host_topology:
            return True

        if fields.CPUAllocationPolicy.DEDICATED not in cpu_policy:
            return True

        if fields.CPUThreadAllocationPolicy.REQUIRE not in cpu_thread_policy:
            return True

        # the presence of siblings in at least one cell indicates
        # hyperthreading (HT)
        has_hyperthreading = any(cell.siblings for cell in host_topology.cells)

        if not has_hyperthreading:
            LOG.debug("%(host_state)s fails CPU policy requirements. "
                      "Host does not have hyperthreading or "
                      "hyperthreading is disabled, but 'require' threads "
                      "policy was requested.", {'host_state': host_state})
            msg = ("Requested threads policy: '%s'; from "
                   "flavor or image is not allowed on "
                   "non-hyperthreaded host"
                   % cpu_thread_policy)
            details = utils.details_append(details, msg)
            return False

        return True

    def host_passes(self, host_state, spec_obj):
        # WRS - disable this filter for non-libvirt hypervisor
        if not utils.is_libvirt_compute(host_state):
            return True

        # TODO(stephenfin): The 'numa_fit_instance_to_host' function has the
        # unfortunate side effect of modifying 'spec_obj.numa_topology' - an
        # InstanceNUMATopology object - by populating the 'cpu_pinning' field.
        # This is rather rude and said function should be reworked to avoid
        # doing this. That's a large, non-backportable cleanup however, so for
        # now we just duplicate spec_obj to prevent changes propagating to
        # future filter calls.
        # Note that we still need to pass the original spec_obj to
        # filter_reject so the error message persists.
        cloned_spec_obj = spec_obj.obj_clone()

        ram_ratio = host_state.ram_allocation_ratio
        cpu_ratio = host_state.cpu_allocation_ratio
        extra_specs = cloned_spec_obj.flavor.extra_specs
        image_props = cloned_spec_obj.image.properties
        requested_topology = cloned_spec_obj.numa_topology
        host_topology, _fmt = hardware.host_topology_and_format_from_host(
                host_state)
        pci_requests = cloned_spec_obj.pci_requests

        if pci_requests:
            pci_requests = pci_requests.requests

        details = utils.details_initialize(details=None)

        if not self._satisfies_cpu_policy(host_state, extra_specs,
                                          image_props, details=details):
            msg = 'Host not useable. ' + ', '.join(details.get('reason', []))
            self.filter_reject(host_state, spec_obj, msg)
            return False

        if requested_topology and host_topology:
            limits = objects.NUMATopologyLimits(
                cpu_allocation_ratio=cpu_ratio,
                ram_allocation_ratio=ram_ratio)

            # WRS: Support strict vs prefer allocation of PCI devices.
            pci_numa_affinity = extra_specs.get('hw:wrs:pci_numa_affinity',
                'strict')
            pci_strict = False if pci_numa_affinity == 'prefer' else True

            # L3 CAT Support
            if any(cell.cachetune_requested
                   for cell in requested_topology.cells):
                free_closids = (host_state.l3_closids -
                                host_state.l3_closids_used)
                if free_closids < 1:
                    msg = ('Insufficient L3 closids: '
                           'req:%(req)s, avail:%(avail)s' %
                           {'req': 1, 'avail': free_closids})
                    self.filter_reject(host_state, spec_obj, msg)
                    return False
                # save limit for compute node to test against
                host_state.limits['closids'] = host_state.l3_closids

            instance_topology = (hardware.numa_fit_instance_to_host(
                        host_topology, requested_topology,
                        limits=limits,
                        pci_requests=pci_requests,
                        pci_stats=host_state.pci_stats,
                        details=details,
                        pci_strict=pci_strict))
            if not instance_topology:
                LOG.debug("%(host)s, %(node)s fails NUMA topology "
                          "requirements. The instance does not fit on this "
                          "host.", {'host': host_state.host,
                                    'node': host_state.nodename},
                          instance_uuid=spec_obj.instance_uuid)
                msg = details.get('reason', [])
                self.filter_reject(host_state, spec_obj, msg)
                return False
            host_state.limits['numa_topology'] = limits
            return True
        elif requested_topology:
            LOG.debug("%(host)s, %(node)s fails NUMA topology requirements. "
                      "No host NUMA topology while the instance specified "
                      "one.",
                      {'host': host_state.host, 'node': host_state.nodename},
                      instance_uuid=spec_obj.instance_uuid)
            msg = 'Missing host topology'
            self.filter_reject(host_state, spec_obj, msg)
            return False
        else:
            return True
