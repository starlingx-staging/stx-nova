# Copyright (c) 2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
SoftwareMgmtWeigher:
- Prefer hosts that are patch current while there remains other hosts that are
  not current.
- Prefer hosts that are upgrades current while upgrades are in progress.
"""
from oslo_config import cfg

from nova.scheduler import weights

CONF = cfg.CONF


class SoftwareMgmtWeigher(weights.BaseHostWeigher):
    minval = 0

    def _weigh_object(self, host_state, weight_properties):
        """Higher weights win. We want to choose the preferred hosts."""

        weight = 0.0
        if host_state.patch_prefer:
            weight += CONF.filter_scheduler.swmgmt_patch_weight_multiplier
        if host_state.upgrade_prefer:
            weight += CONF.filter_scheduler.swmgmt_upgrade_weight_multiplier
        return weight
