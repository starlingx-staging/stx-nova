# Copyright 2010 OpenStack Foundation
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

from itertools import islice
import six
import webob

from nova.api.openstack import common
from nova.api.openstack.compute.schemas import flavors_extraspecs
from nova.api.openstack import extensions
from nova.api.openstack import wsgi
from nova.api import validation
from nova import context as novacontext
from nova import exception
from nova.i18n import _
from nova import objects
from nova.objects import fields
from nova.objects import image_meta
from nova.policies import flavor_extra_specs as fes_policies
from nova import utils
from nova.virt import hardware

from oslo_serialization import jsonutils
from oslo_utils import strutils

# flavor extra specs keys needed for multiple validation routines
CPU_POLICY_KEY = 'hw:cpu_policy'
CPU_SCALING_KEY = 'hw:wrs:min_vcpus'
SHARED_VCPU_KEY = 'hw:wrs:shared_vcpu'

# host numa nodes
MAX_HOST_NUMA_NODES = 4


class FlavorExtraSpecsController(wsgi.Controller):
    """The flavor extra specs API controller for the OpenStack API."""

    def _check_flavor_in_use(self, flavor):
        if flavor.is_in_use():
            msg = _('Updating extra specs not permitted when flavor is '
                    'associated to one or more valid instances')
            raise webob.exc.HTTPBadRequest(explanation=msg)

    @staticmethod
    def _validate_vcpu_models(flavor):
        key = 'hw:cpu_model'
        if key in flavor.extra_specs:
            model = flavor.extra_specs[key]
            if model not in fields.CPUModel.ALL:
                msg = (_("Invalid %(K)s '%(M)s', must be one of: %(V)s.") %
                       {'K': key,
                        'M': model,
                        'V': ', '.join(fields.CPUModel.ALL)
                       })
                raise webob.exc.HTTPBadRequest(explanation=msg)

    @staticmethod
    def _validate_numa_node(flavor):
        NUMA_NODES_KEY = 'hw:numa_nodes'
        NUMA_NODE_PREFIX = 'hw:numa_node.'
        specs = flavor.extra_specs
        try:
            hw_numa_nodes = int(specs.get(NUMA_NODES_KEY, 1))
        except ValueError:
            msg = _('hw:numa_nodes value must be an integer')
            raise webob.exc.HTTPBadRequest(explanation=msg)
        if hw_numa_nodes < 1:
            msg = _('hw:numa_nodes value must be greater than 0')
            raise webob.exc.HTTPBadRequest(explanation=msg)

        for key in specs:
            if key.startswith(NUMA_NODE_PREFIX):
                # NUMA pinning not allowed when CPU policy is shared
                if (specs.get(CPU_POLICY_KEY) ==
                        fields.CPUAllocationPolicy.SHARED):
                    msg = _('hw:numa_node not permitted when cpu policy '
                            'is set to shared')
                    raise webob.exc.HTTPConflict(explanation=msg)
                suffix = key.split(NUMA_NODE_PREFIX, 1)[1]
                try:
                    vnode = int(suffix)
                except ValueError:
                    msg = _('virtual numa node number must be an integer')
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                if vnode < 0:
                    msg = _('virtual numa node number must be greater than or '
                            'equal to 0')
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                try:
                    pnode = int(specs[key])
                except ValueError:
                    msg = _('%s must be an integer') % key
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                if pnode < 0:
                    msg = _('%s must be greater than or equal to 0') % key
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                if pnode >= MAX_HOST_NUMA_NODES:
                    msg = (_('%(K)s value %(P)d is not valid. It must '
                             'be an integer from 0 to %(N)d') %
                           {'K': key,
                            'P': pnode,
                            'N': MAX_HOST_NUMA_NODES - 1
                           })
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                if vnode >= hw_numa_nodes:
                    msg = _('all hw:numa_node keys must use vnode id less than'
                            ' the specified hw:numa_nodes value (%s)') \
                        % hw_numa_nodes
                    raise webob.exc.HTTPBadRequest(explanation=msg)

        # CPU scaling doesn't currently support multiple guest NUMA nodes
        if hw_numa_nodes > 1 and CPU_SCALING_KEY in specs:
            msg = _('CPU scaling not supported for instances with'
                    ' multiple NUMA nodes.')
            raise webob.exc.HTTPConflict(explanation=msg)

        # CGTS-3716 Asymmetric NUMA topology protection
        # Do common error check from numa_get_constraints with a clearer error
        if hw_numa_nodes > 0 and specs.get('hw:numa_cpus.0') is None:
            if (flavor.vcpus % hw_numa_nodes) > 0:
                msg = _('flavor vcpus not evenly divisible by'
                        ' the specified hw:numa_nodes value (%s)') \
                      % hw_numa_nodes
                raise webob.exc.HTTPConflict(explanation=msg)
            if (flavor.memory_mb % hw_numa_nodes) > 0:
                msg = _('flavor memory not evenly divisible by'
                        ' the specified hw:numa_nodes value (%s) so'
                        ' per NUMA-node values must be explicitly specified') \
                      % hw_numa_nodes
                raise webob.exc.HTTPConflict(explanation=msg)

        # Catchall test
        try:
            # Check if this modified flavor would be valid assuming
            # no image metadata.
            hardware.numa_get_constraints(flavor, image_meta.ImageMeta(
                    properties=image_meta.ImageMetaProps()))
        except Exception as error:
            msg = _('%s') % error.message
            raise webob.exc.HTTPConflict(explanation=msg)

    @staticmethod
    def _validate_cache_node(flavor):
        # Split a list into evenly sized chunks
        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        specs = flavor.extra_specs
        KEYS = ['hw:cache_l3', 'hw:cache_l3_code', 'hw:cache_l3_data']
        for this_key in KEYS:
            this_prefix = this_key + '.'
            for key in specs:
                if key.startswith(this_prefix):
                    # Check that we have specified dedicated cpus
                    if specs.get(CPU_POLICY_KEY) != \
                            fields.CPUAllocationPolicy.DEDICATED:
                        msg = (_('%(K)s is not permitted when %(P)s is set to'
                                 ' shared.') %
                               {'K': this_key,
                                'P': CPU_POLICY_KEY})
                        raise webob.exc.HTTPConflict(explanation=msg)

                    # Virtual numa node must be valid
                    suffix = key.split(this_prefix, 1)[1]
                    try:
                        vnode = int(suffix)
                    except ValueError:
                        msg = _('%s virtual numa node number must be an '
                                'integer') % this_key
                        raise webob.exc.HTTPBadRequest(explanation=msg)
                    if vnode < 0:
                        msg = _('%s virtual numa node number must be greater '
                                'than or equal to 0') % this_key
                        raise webob.exc.HTTPBadRequest(explanation=msg)

                    # Cache size must be valid and positive
                    try:
                        value = int(specs[key])
                    except ValueError:
                        msg = _('%s must be an integer') % key
                        raise webob.exc.HTTPBadRequest(explanation=msg)
                    if value <= 0:
                        msg = _('%s must be positive') % key
                        raise webob.exc.HTTPBadRequest(explanation=msg)

        # Check that we can properly parse hw:cache_vcpus.x cpulist,
        # and that vcpus are within valid range.
        flavor_cpuset = set(range(flavor.vcpus))
        cache_key = 'hw:cache_vcpus'
        cache_prefix = cache_key + '.'
        for key in specs:
            if key.startswith(cache_prefix):
                suffix = key.split(cache_prefix, 1)[1]
                try:
                    vnode = int(suffix)
                except ValueError:
                    msg = _('%s virtual numa node number must be an '
                            'integer') % cache_key
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                if vnode < 0:
                    msg = _('%s virtual numa node number must be greater '
                            'than or equal to 0') % cache_key
                    raise webob.exc.HTTPBadRequest(explanation=msg)

                try:
                    value = specs[key]
                    cpuset_ids = hardware.parse_cpu_spec(value)
                except Exception as e:
                    msg = (_("Invalid %(K)s '%(V)s'; reason: %(R)s.") %
                           {'K': key,
                            'V': value,
                            'R': e.format_message()
                            })
                    raise webob.exc.HTTPBadRequest(explanation=msg)

                if not cpuset_ids:
                    msg = (_("Invalid %(K)s '%(V)s'; reason: %(R)s.") %
                           {'K': key,
                            'V': value,
                            'R': 'no vcpus specified'
                            })
                    raise webob.exc.HTTPBadRequest(explanation=msg)

                if not cpuset_ids.issubset(flavor_cpuset):
                    msg = _('%(K)s value (%(V)s) must be a subset of vcpus '
                            '(%(S)s)') \
                            % {'K': key, 'V': value,
                               'S': utils.list_to_range(list(flavor_cpuset))}
                    raise webob.exc.HTTPBadRequest(explanation=msg)

                # Check whether hw:cache_vcpus.x are subset of hw:numa_cpus.x
                cpus_key = 'hw:numa_cpus.' + suffix
                if cpus_key in specs:
                    try:
                        cpus_value = specs[cpus_key]
                        numa_cpuset = hardware.parse_cpu_spec(cpus_value)
                    except Exception as e:
                        msg = (_("Invalid %(K)s '%(V)s'; reason: %(R)s.") %
                               {'K': cpus_key,
                                'V': cpus_value,
                                'R': e.format_message()
                                })
                        raise webob.exc.HTTPBadRequest(explanation=msg)
                else:
                    NUMA_NODES_KEY = 'hw:numa_nodes'
                    try:
                        hw_numa_nodes = int(specs.get(NUMA_NODES_KEY, 1))
                    except ValueError:
                        msg = _('hw:numa_nodes value must be an integer')
                        raise webob.exc.HTTPBadRequest(explanation=msg)
                    if vnode >= hw_numa_nodes:
                        msg = (_('%(K)s must use vnode id less than the '
                                 'specified hw:numa_nodes value %(N)s.') %
                               {'K': this_key,
                                'N': hw_numa_nodes})
                        raise webob.exc.HTTPBadRequest(explanation=msg)
                    chunk_size = flavor.vcpus / hw_numa_nodes
                    numa_cpus = list(chunk(range(flavor.vcpus), chunk_size))
                    try:
                        numa_cpuset = set(numa_cpus[vnode])
                    except IndexError:
                        msg = _('%s virtual numa node number must be subset '
                                'of numa nodes') % vnode
                        raise webob.exc.HTTPBadRequest(explanation=msg)
                if not cpuset_ids.issubset(numa_cpuset):
                    msg = (_('%(K)s value (%(V)s) must be a subset of '
                             'vcpus (%(S)s)') %
                           {'K': cache_key, 'V': value,
                            'S': utils.list_to_range(list(numa_cpuset))
                            })
                    raise webob.exc.HTTPBadRequest(explanation=msg)

                # Check that we have specified dedicated cpus
                if specs.get(CPU_POLICY_KEY) != \
                        fields.CPUAllocationPolicy.DEDICATED:
                    msg = (_('%(K)s is not permitted when %(P)s is set to'
                             ' shared.') %
                           {'K': key,
                            'P': CPU_POLICY_KEY})
                    raise webob.exc.HTTPConflict(explanation=msg)

    # WRS: Validate pci numa affinity.
    @staticmethod
    def _validate_pci_numa_affinity(flavor):
        key = 'hw:wrs:pci_numa_affinity'
        if key in flavor.extra_specs:
            value = flavor.extra_specs[key]
            if value not in fields.PciAllocationPolicy.ALL:
                msg = (_("Invalid %(K)s '%(V)s', must be one of: %(A)s.") %
                       {'K': key,
                        'V': value,
                        'A': ', '.join(
                            list(fields.PciAllocationPolicy.ALL))
                        })
                raise webob.exc.HTTPBadRequest(explanation=msg)

    # WRS: Validate pci irq affinity mask
    @staticmethod
    def _validate_pci_irq_affinity_mask(flavor):
        key = 'hw:pci_irq_affinity_mask'
        if key in flavor.extra_specs:
            value = flavor.extra_specs[key]

            # check that we can properly parse the mask
            try:
                cpuset_ids = hardware._get_pci_affinity_mask(flavor)
            except Exception as e:
                msg = (_("Invalid %(K)s '%(V)s'; reason: %(R)s.") %
                       {'K': key,
                        'V': value,
                        'R': e.format_message()
                        })
                raise webob.exc.HTTPBadRequest(explanation=msg)

            # check that cpuset_ids are within valid range
            flavor_cpuset = set(range(flavor.vcpus))
            if not cpuset_ids.issubset(flavor_cpuset):
                msg = _('%(K)s value (%(V)s) must be a subset of vcpus '
                        '(%(S)s)') \
                        % {'K': key, 'V': value,
                           'S': utils.list_to_range(list(flavor_cpuset))}
                raise webob.exc.HTTPBadRequest(explanation=msg)

            # Check that we have specified dedicated cpus
            if flavor.extra_specs.get(CPU_POLICY_KEY) != \
                    fields.CPUAllocationPolicy.DEDICATED:
                msg = _('%(K)s is only valid when %(P)s is %(D)s.  Either '
                        'set extra spec %(P)s to %(D)s or do not set %(K)s.') \
                      % {'K': key,
                         'P': CPU_POLICY_KEY,
                         'D': fields.CPUAllocationPolicy.DEDICATED}
                raise webob.exc.HTTPConflict(explanation=msg)

    # WRS: Validate hw:cpu_realtime_mask and interaction with
    # hw:wrs:shared_vcpu .
    @staticmethod
    def _validate_cpu_realtime_mask(flavor):
        key = 'hw:cpu_realtime_mask'
        if key in flavor.extra_specs:
            value = flavor.extra_specs[key]

            # Check we can properly parse the mask, that the mask defines both
            # realtime and normal vCPUs, and that shared_vcpu is a subset of
            # normal vCPUs.
            try:
                image = objects.ImageMeta.from_dict({"properties": {}})
                hardware.vcpus_realtime_topology(flavor, image)
            except Exception as e:
                msg = (_("Invalid %(K)s '%(V)s', reason: %(R)s.") %
                       {'K': key,
                        'V': value,
                        'R': e.format_message()
                        })
                raise webob.exc.HTTPBadRequest(explanation=msg)

    # WRS: cpu policy extra spec
    @staticmethod
    def _validate_cpu_policy(flavor):
        key = CPU_POLICY_KEY
        specs = flavor.extra_specs
        if key in specs:
            value = specs[key]
            if value not in fields.CPUAllocationPolicy.ALL:
                msg = _("invalid %(K)s '%(V)s', must be one of: %(A)s") \
                        % {'K': key,
                           'V': value,
                           'A': ', '.join(
                            list(fields.CPUAllocationPolicy.ALL))}
                raise webob.exc.HTTPBadRequest(explanation=msg)

    # WRS: cpu thread policy extra spec
    @staticmethod
    def _validate_cpu_thread_policy(flavor):
        key = 'hw:cpu_thread_policy'
        specs = flavor.extra_specs
        if key in specs:
            value = specs[key]
            if value not in fields.CPUThreadAllocationPolicy.ALL:
                msg = _("invalid %(K)s '%(V)s', must be one of %(A)s") \
                        % {'K': key,
                           'V': value,
                           'A': ', '.join(
                            list(fields.CPUThreadAllocationPolicy.ALL))}
                raise webob.exc.HTTPBadRequest(explanation=msg)
            if specs.get(CPU_POLICY_KEY) != \
                    fields.CPUAllocationPolicy.DEDICATED:
                msg = _('%(K)s is only valid when %(P)s is %(D)s.  Either '
                        'unset %(K)s or set %(P)s to %(D)s.') \
                        % {'K': key,
                           'P': CPU_POLICY_KEY,
                           'D': fields.CPUAllocationPolicy.DEDICATED}
                raise webob.exc.HTTPConflict(explanation=msg)
            if SHARED_VCPU_KEY in specs:
                if value in [fields.CPUThreadAllocationPolicy.REQUIRE,
                             fields.CPUThreadAllocationPolicy.ISOLATE]:
                    msg = _('Cannot set %(K)s to %(V)s if %(S)s is set. '
                            'Either unset %(K)s, set it to %(P)s, or unset '
                            '%(S)s') \
                            % {'K': key,
                               'V': value,
                               'S': SHARED_VCPU_KEY,
                               'P': fields.CPUThreadAllocationPolicy.PREFER}
                    raise webob.exc.HTTPConflict(explanation=msg)
            if CPU_SCALING_KEY in specs:
                if value == fields.CPUThreadAllocationPolicy.REQUIRE:
                    msg = _('Cannot set %(K)s to %(V)s if %(C)s is set. '
                             'Either unset %(K)s, set it to another policy, '
                             'or unset %(C)s') \
                            % {'K': key,
                               'V': value,
                               'C': CPU_SCALING_KEY}
                    raise webob.exc.HTTPConflict(explanation=msg)

    # WRS: check that we do not use deprecated WRS keys
    @staticmethod
    def _validate_wrs_deprecated_keys(flavor):
        keys = ['hw:wrs:vcpu:scheduler']

        specs = flavor.extra_specs
        for key in keys:
            if key in specs:
                msg = _('%(k)s is no longer supported.') % {'k': key}
                raise webob.exc.HTTPBadRequest(explanation=msg)

    @staticmethod
    def _validate_min_vcpus(flavor):
        key = CPU_SCALING_KEY
        specs = flavor.extra_specs
        if key in specs:
            if specs.get(CPU_POLICY_KEY) != \
                    fields.CPUAllocationPolicy.DEDICATED:
                msg = "%s is only valid when %s is %s." % \
                      (key, CPU_POLICY_KEY,
                            fields.CPUAllocationPolicy.DEDICATED)
                raise webob.exc.HTTPConflict(explanation=msg)
            try:
                min_vcpus = int(specs[key])
            except ValueError:
                msg = _('%s must be an integer') % key
                raise webob.exc.HTTPBadRequest(explanation=msg)
            if min_vcpus < 1:
                msg = _('%s must be greater than or equal to 1') % key
                raise webob.exc.HTTPBadRequest(explanation=msg)
            if min_vcpus > flavor.vcpus:
                msg = _('%(K)s must be less than or equal to the '
                        'flavor vcpus value of %(V)d') % \
                        {'K': key, 'V': flavor.vcpus}
                raise webob.exc.HTTPConflict(explanation=msg)

    # WRS: extra spec sw:wrs keys validation
    @staticmethod
    def _validate_sw_keys(flavor):
        keys = ['sw:wrs:auto_recovery', 'sw:wrs:guest:heartbeat',
                'sw:wrs:vtpm', 'hw:hpet']
        specs = flavor.extra_specs
        for key in keys:
            if key in specs:
                value = specs[key].lower()
                if value not in ('false', 'true'):
                    msg = _('%(k)s must be True or False, value: %(v)s') \
                             % {'k': key, 'v': value}
                    raise webob.exc.HTTPBadRequest(explanation=msg)

    @staticmethod
    def _validate_nested_vmx(flavor):
        key = 'hw:wrs:nested_vmx'
        specs = flavor.extra_specs
        if key in specs:
            value = specs[key]
            try:
                is_vmx_requested = strutils.bool_from_string(value,
                                                             strict=True)
            except ValueError as error:
                raise webob.exc.HTTPBadRequest(explanation=error.message)

            # Check if at least one Host has 'vmx' feature enabled
            if is_vmx_requested:
                context = novacontext.get_admin_context()
                compute_nodes = objects.ComputeNodeList.get_all(context)
                for node in compute_nodes:
                    cpu_info = jsonutils.loads(node.cpu_info)
                    if 'vmx' in cpu_info['features']:
                        return
                msg = _("No Compute host was found with vmx enabled.")
                raise webob.exc.HTTPConflict(explanation=msg)

    # WRS: shared vcpu extra spec
    @staticmethod
    def _validate_shared_vcpu(flavor):
        key = SHARED_VCPU_KEY
        specs = flavor.extra_specs
        if key in specs:
            try:
                value = int(specs[key])
            except ValueError:
                msg = _('%s must be an integer') % key
                raise webob.exc.HTTPBadRequest(explanation=msg)
            if value < 0:
                msg = _('%s must be greater than or equal to 0') % key
                raise webob.exc.HTTPBadRequest(explanation=msg)
            if value >= flavor.vcpus:
                msg = _('%(K)s value (%(V)d) must be less than flavor vcpus '
                        '(%(F)d)') \
                        % {'K': key, 'V': value, 'F': flavor.vcpus}
                raise webob.exc.HTTPBadRequest(explanation=msg)
            if specs.get(CPU_POLICY_KEY) != \
                    fields.CPUAllocationPolicy.DEDICATED:
                msg = _('%(K)s is only valid when %(P)s is %(D)s.  Either '
                       'set extra spec %(P)s to %(D)s or do not set %(K)s.') \
                        % {'K': key,
                           'P': CPU_POLICY_KEY,
                           'D': fields.CPUAllocationPolicy.DEDICATED}
                raise webob.exc.HTTPConflict(explanation=msg)
            if (value != 0) and (CPU_SCALING_KEY in specs):
                msg = _('%(SH)s value (%(V)d) is incompatible '
                        'with %(SC)s. %(SH)s may only be 0 when %(SC)s '
                        'is specified.') \
                        % {'SH': SHARED_VCPU_KEY,
                           'SC': CPU_SCALING_KEY,
                           'V': value}
                raise webob.exc.HTTPConflict(explanation=msg)

    # WRS: Should this go in the flavor object as part of the save()
    # routine?  If you really need the context to validate something,
    # add it back in to the args.
    def _validate_extra_specs(self, flavor):
        self._validate_vcpu_models(flavor)
        self._validate_cpu_policy(flavor)
        self._validate_cpu_thread_policy(flavor)
        self._validate_pci_numa_affinity(flavor)
        self._validate_pci_irq_affinity_mask(flavor)
        self._validate_shared_vcpu(flavor)
        self._validate_min_vcpus(flavor)
        self._validate_numa_node(flavor)
        self._validate_cache_node(flavor)
        common.validate_live_migration_timeout(flavor.extra_specs)
        common.validate_live_migration_max_downtime(flavor.extra_specs)
        self._validate_sw_keys(flavor)
        self._validate_nested_vmx(flavor)
        common.validate_boolean_options(flavor.extra_specs)
        self._validate_cpu_realtime_mask(flavor)
        self._validate_wrs_deprecated_keys(flavor)

    def _get_extra_specs(self, context, flavor_id):
        flavor = common.get_flavor(context, flavor_id)
        return dict(extra_specs=flavor.extra_specs)

    # NOTE(gmann): Max length for numeric value is being checked
    # explicitly as json schema cannot have max length check for numeric value
    def _check_extra_specs_value(self, specs):
        for value in specs.values():
            try:
                if isinstance(value, (six.integer_types, float)):
                    value = six.text_type(value)
                    utils.check_string_length(value, 'extra_specs value',
                                              max_length=255)
            except exception.InvalidInput as error:
                raise webob.exc.HTTPBadRequest(
                          explanation=error.format_message())

    @extensions.expected_errors(404)
    def index(self, req, flavor_id):
        """Returns the list of extra specs for a given flavor."""
        context = req.environ['nova.context']
        context.can(fes_policies.POLICY_ROOT % 'index')
        return self._get_extra_specs(context, flavor_id)

    # NOTE(gmann): Here should be 201 instead of 200 by v2.1
    # +microversions because the flavor extra specs has been created
    # completely when returning a response.
    @extensions.block_during_upgrade()
    @extensions.expected_errors((400, 404, 409))
    @validation.schema(flavors_extraspecs.create)
    def create(self, req, flavor_id, body):
        context = req.environ['nova.context']
        context.can(fes_policies.POLICY_ROOT % 'create')

        specs = body['extra_specs']
        self._check_extra_specs_value(specs)
        flavor = common.get_flavor(context, flavor_id)
        self._check_flavor_in_use(flavor)
        try:
            flavor.extra_specs = dict(flavor.extra_specs, **specs)
            self._validate_extra_specs(flavor)
            flavor.save()
        except exception.FlavorExtraSpecUpdateCreateFailed as e:
            raise webob.exc.HTTPConflict(explanation=e.format_message())
        except exception.FlavorNotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.format_message())
        return body

    @extensions.block_during_upgrade()
    @extensions.expected_errors((400, 404, 409))
    @validation.schema(flavors_extraspecs.update)
    def update(self, req, flavor_id, id, body):
        context = req.environ['nova.context']
        context.can(fes_policies.POLICY_ROOT % 'update')

        self._check_extra_specs_value(body)
        if id not in body:
            expl = _('Request body and URI mismatch')
            raise webob.exc.HTTPBadRequest(explanation=expl)
        flavor = common.get_flavor(context, flavor_id)
        self._check_flavor_in_use(flavor)
        try:
            flavor.extra_specs = dict(flavor.extra_specs, **body)
            self._validate_extra_specs(flavor)
            flavor.save()
        except exception.FlavorExtraSpecUpdateCreateFailed as e:
            raise webob.exc.HTTPConflict(explanation=e.format_message())
        except exception.FlavorNotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.format_message())
        return body

    @extensions.expected_errors(404)
    def show(self, req, flavor_id, id):
        """Return a single extra spec item."""
        context = req.environ['nova.context']
        context.can(fes_policies.POLICY_ROOT % 'show')
        flavor = common.get_flavor(context, flavor_id)
        try:
            return {id: flavor.extra_specs[id]}
        except KeyError:
            msg = _("Flavor %(flavor_id)s has no extra specs with "
                    "key %(key)s.") % dict(flavor_id=flavor_id,
                                           key=id)
            raise webob.exc.HTTPNotFound(explanation=msg)

    # NOTE(gmann): Here should be 204(No Content) instead of 200 by v2.1
    # +microversions because the flavor extra specs has been deleted
    # completely when returning a response.
    @extensions.block_during_upgrade()
    @extensions.expected_errors((400, 404, 409))
    def delete(self, req, flavor_id, id):
        """Deletes an existing extra spec."""
        context = req.environ['nova.context']
        context.can(fes_policies.POLICY_ROOT % 'delete')
        flavor = common.get_flavor(context, flavor_id)
        self._check_flavor_in_use(flavor)
        try:
            # The id object is an aggregation of multiple extra spec keys
            # The keys are aggregated using the ';'  character
            # This allows multiple extra specs to be deleted in one call
            # This is required since some validators will raise an exception
            # if one extra spec exists while another is missing
            ids = id.split(';')
            for an_id in ids:
                del flavor.extra_specs[an_id]
            self._validate_extra_specs(flavor)
            flavor.save()
        except (exception.FlavorExtraSpecsNotFound,
                exception.FlavorNotFound) as e:
            raise webob.exc.HTTPNotFound(explanation=e.format_message())
        except KeyError:
            msg = _("Flavor %(flavor_id)s has no extra specs with "
                    "key %(key)s.") % dict(flavor_id=flavor_id,
                                           key=id)
            raise webob.exc.HTTPNotFound(explanation=msg)
