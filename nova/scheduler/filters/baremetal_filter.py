# Copyright (c) 2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

from oslo_log import log as logging
from oslo_utils import strutils

from nova.scheduler import filters
from nova import utils as nova_utils


BAREMETAL_KEY = 'baremetal'
BAREMETAL_VALUE = 'true'

LOG = logging.getLogger(__name__)


class BaremetalFilter(filters.BaseHostFilter):
    """Filter hosts that support baremetal if specified.
    """

    @staticmethod
    def is_baremetal_enabled(flavor):
        flavor_baremetal = flavor.get('extra_specs', {}).get(BAREMETAL_KEY)
        return strutils.bool_from_string(flavor_baremetal)

    def host_passes(self, host_state, spec_obj):
        """Filter based on host hypervisor_type and baremetal extra-spec.

         Host is capable if:
         - hypervisor_type is 'ironic' and 'baremetal=true', OR,
         - hypervisor_type is not 'ironic' and baremetal not specified
           (or 'baremetal=false').
        """
        is_ironic = nova_utils.is_ironic_compute(host_state)
        is_baremetal = self.is_baremetal_enabled(spec_obj.flavor)

        if is_ironic:
            if not is_baremetal:
                msg = ("hypervisor '%(hv)s' requires extra_specs "
                       "%(k)s=%(v)s." %
                       {'hv': host_state.hypervisor_type,
                        'k': BAREMETAL_KEY, 'v': BAREMETAL_VALUE})
                self.filter_reject(host_state, spec_obj, msg)
                return False
        else:
            if is_baremetal:
                msg = ("hypervisor '%(hv)s' does not support extra_specs "
                       "%(k)s=%(v)s." %
                       {'hv': host_state.hypervisor_type,
                        'k': BAREMETAL_KEY, 'v': BAREMETAL_VALUE})
                self.filter_reject(host_state, spec_obj, msg)
                return False
        return True
