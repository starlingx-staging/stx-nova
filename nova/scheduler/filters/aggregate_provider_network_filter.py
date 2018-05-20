#
# Copyright (c) 2014-2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Scheduler filter "AggregateProviderNetworkFilter", host_passes() returns True
when the host is a member of host-aggregate with metadata key
provider:physical_network and contains a superset of physical network values
required by each tenant network specified by the current instance.
"""

from oslo_log import log as logging

from nova.i18n import _LI
from nova.scheduler import filters

from nova.scheduler.filters import utils
from nova import utils as nova_utils

LOG = logging.getLogger(__name__)


class AggregateProviderNetworkFilter(filters.BaseHostFilter):
    """Filter hosts that have the necessary provider physical network(s)."""

    # Aggregate data and tenant do not change within a request
    run_filter_once_per_request = True

    def host_passes(self, host_state, spec_obj):
        """If the host is in an aggregate with metadata key
        "provider:physical_network" and contains the set of values
        needed by each tenant network, it may create instances.
        """

        # WRS - disable this filter for ironic hypervisor
        if nova_utils.is_ironic_compute(host_state):
            return True

        physkey = 'provider:physical_network'
        scheduler_hints = {}
        if spec_obj.obj_attr_is_set('scheduler_hints'):
            scheduler_hints = spec_obj.scheduler_hints or {}
        physnets = set(scheduler_hints.get(physkey, []))
        metadata = utils.aggregate_metadata_get_by_host(host_state,
                                                        key=physkey)
        # Match each provider physical network with host-aggregate metadata.
        if metadata:
            if not physnets.issubset(metadata[physkey]):
                msg = ("%s = %r, require: %r"
                       % (str(physkey),
                          list(metadata[physkey]),
                          list(physnets)))
                self.filter_reject(host_state, spec_obj, msg)
                return False
            else:
                LOG.info(_LI("(%(host)s, %(nodename)s) PASS. "
                             "%(key)s = %(metalist)r. "
                             "require: %(physnetlist)r"),
                         {'host': host_state.host,
                          'nodename': host_state.nodename,
                          'key': physkey,
                          'metalist': list(metadata[physkey]),
                          'physnetlist': list(physnets)})
        else:
            LOG.info(_LI("(%(host)s, %(nodename)s) NOT CONFIGURED. "
                         "%(key)s = %(metalist)r. "
                         "require: %(physnetlist)r"),
                         {'host': host_state.host,
                          'nodename': host_state.nodename,
                          'key': physkey,
                          'metalist': metadata,
                          'physnetlist': list(physnets)})
            return False
        return True
