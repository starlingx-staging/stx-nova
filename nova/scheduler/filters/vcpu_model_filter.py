#
# Copyright (c) 2013-2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Scheduler filter "VCpuModelFilter", host_passes() returns True when the host
CPU model is newer than or equal to the guest CPU model as specified in the
instance type.

The CPU models considered are currently limited to a subset of the Intel CPU
family.  See nova.constants.VCPU_MODEL for the complete list.
"""

from nova import context
from nova import exception
from nova import objects
from nova.objects import fields
from nova.scheduler import filters
from oslo_log import log as logging
from oslo_serialization import jsonutils

LOG = logging.getLogger(__name__)


class VCpuModelFilter(filters.BaseHostFilter):
    """Filter hosts that support the necessary virtual cpu model.
    """

    # Instance type data does not change within a request
    run_filter_once_per_request = True

    def _is_compatible(self, host, guest):
        """Determine if the host CPU model is capable of emulating the guest CPU
        model.  For this determination we are currently only interested in
        Intel based CPUs in the most recent family (from pentiumpro to
        Haswell)
        """
        try:
            host_index = fields.CPUModel.ALL.index(host)
        except ValueError:
            # The host CPU model is not in our list.  This is unexpected and
            # likely indicates that support for a new processor needs to be
            # added and validated.
            return False
        try:
            guest_index = fields.CPUModel.ALL.index(guest)
        except ValueError:
            # The guest CPU model is not in our list.  We can't tell whether
            # we can support it or not.
            return False

        # if guest request IBRS cpu, but host does not support 'IBRS', then
        # return false directly
        if('IBRS' in guest and 'IBRS' not in host):
            return False

        return bool(guest_index <= host_index)

    def _passthrough_host_passes(self, host_state, spec_obj):
        hints = spec_obj.scheduler_hints
        if not hints['host'] or not hints['node']:
            LOG.info("(%(host)s, %(node)s) CANNOT SCHEDULE: "
                     "VCPU Passthrough guest migrating from unknown host",
                     {'host': host_state.host,
                     'node': host_state.nodename})
            msg = ("Passthrough cpu model migrating from unknown host")
            self.filter_reject(host_state, spec_obj, msg)
            return False

        ctxt = context.get_admin_context()
        try:
            source_node = objects.ComputeNode.get_by_host_and_nodename(ctxt,
                hints['host'][0], hints['node'][0])
        except exception.NotFound:
            LOG.info("(%(host)s, %(node)s) CANNOT SCHEDULE: "
                     "No compute node record found for source %(h)s, %(n)s)",
                     {'host': host_state.host,
                     'node': host_state.nodename,
                     'h': hints['host'][0], 'n': hints['node'][0]})
            msg = ("No compute node record found for %(host)s, %(node)s)" %
                   {'host': hints['host'][0], 'node': hints['node'][0]})
            self.filter_reject(host_state, spec_obj, msg)
            return False

        source_model = self._get_cpu_model(source_node.cpu_info)
        host_model = self._get_cpu_model(host_state.cpu_info)
        same_cpu_features = self._cpus_have_same_features(host_state.cpu_info,
                                                      source_node.cpu_info)
        key = 'vcpu_model'
        if source_model != host_model or not same_cpu_features:
            LOG.info("(%(host)s, %(node)s) CANNOT SCHEDULE: "
                     "different model or incompatible cpu features: "
                     "host %(key)s = %(host_model)s, "
                     "required = %(required)s",
                     {'host': host_state.host,
                     'node': host_state.nodename,
                     'key': key,
                     'host_model': host_model,
                     'required': source_model})
            msg = ("Different VCPU model or cpu features. "
                   "Host %(host_model)s required %(req)s" %
                   {'host_model': host_model, 'req': source_model})
            self.filter_reject(host_state, spec_obj, msg)
            return False

        LOG.info("(%(host)s, %(node)s) "
                 "PASS: host %(key)s = %(host_model)s, "
                 "required = %(required)s",
                 {'host': host_state.host,
                 'node': host_state.nodename,
                 'key': key,
                 'host_model': host_model,
                 'required': source_model})
        return True

    def _get_cpu_model(self, cpu_info):
        """Parses the driver CPU info structure to extract the host CPU model
        """
        info = jsonutils.loads(cpu_info)
        return info['model']

    def _cpus_have_same_features(self, host_cpu_info, cpu_info):
        info = jsonutils.loads(cpu_info)
        host_info = jsonutils.loads(host_cpu_info)
        return cmp(info['features'], host_info['features']) == 0

    def _is_host_kvm(self, cpu_info):
        info = jsonutils.loads(cpu_info)
        if 'vmx' in info['features']:
            return True
        return False

    def host_passes(self, host_state, spec_obj):
        """If the host CPU model is newer than or equal to the guest CPU model
        that is specified in the flavor or image then this host is deemed
        capable of instantiating this instance.
        """
        flavor_model = spec_obj.flavor.extra_specs.get("hw:cpu_model")
        image_model = spec_obj.image.properties.get("hw_cpu_model")
        if (image_model is not None and flavor_model is not None and
            image_model != flavor_model):
            raise exception.ImageVCPUModelForbidden()

        model = flavor_model or image_model
        if not model:
            LOG.debug("(%(host)s, %(node)s) PASS: no required vCPU model",
                     {'host': host_state.host,
                     'node': host_state.nodename})
            return True

        if model == 'Passthrough' and \
                not self._is_host_kvm(host_state.cpu_info):
            LOG.info("(%(host)s, %(node)s) CANNOT SCHEDULE: "
                     "Passthrough VCPU Model only available on 'kvm' hosts",
                     {'host': host_state.host,
                     'node': host_state.nodename})
            msg = "Passthrough VCPU Model only available on 'kvm' hosts"
            self.filter_reject(host_state, spec_obj, msg)
            return False

        task_state = spec_obj.scheduler_hints.get('task_state')
        if model == 'Passthrough':
            if task_state and ('scheduling' not in task_state):
                return self._passthrough_host_passes(host_state, spec_obj)

        key = 'vcpu_model'
        host_model = self._get_cpu_model(host_state.cpu_info)
        if self._is_compatible(host_model, model):
            LOG.info("(%(host)s, %(node)s) "
                     "PASS: host %(key)s = %(host_model)s, "
                     "required = %(required)s",
                     {'host': host_state.host,
                     'node': host_state.nodename,
                     'key': key,
                     'host_model': host_model,
                     'required': model})
            return True
        else:
            LOG.info("(%(host)s, %(node)s) CANNOT SCHEDULE: "
                     "host %(key)s = %(host_model)s, "
                     "required = %(required)s",
                     {'host': host_state.host,
                     'node': host_state.nodename,
                     'key': key,
                     'host_model': host_model,
                     'required': model})
            msg = ("Host VCPU model %(host_model)s required %(required)s" %
                   {'host_model': host_model, 'required': model})
            self.filter_reject(host_state, spec_obj, msg)
            return False
