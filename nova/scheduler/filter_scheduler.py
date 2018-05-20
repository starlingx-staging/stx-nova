# Copyright (c) 2011 OpenStack Foundation
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
# Copyright (c) 2016-2017 Wind River Systems, Inc.
#

"""
The FilterScheduler is for creating instances locally.
You can customize this scheduler by specifying your own Host Filters and
Weighing Functions.
"""

import eventlet
import itertools
import pprint
import random

from oslo_log import log as logging
from oslo_utils import strutils
from six.moves import range

import nova.conf
from nova.i18n import _
from nova import rpc
from nova.scheduler import client
from nova.scheduler import driver
from nova.scheduler import utils as scheduler_utils

# WRS - network provider filter
from nova import context as novacontext
from nova import exception
from nova import network
from nova import objects
from nova.pci import utils as pci_utils
from nova import utils

CONF = nova.conf.CONF
LOG = logging.getLogger(__name__)
_query_timeout_seconds = 15


def mround(x, base=5):
    """Round value x to nearest multiple of base."""
    return int(base * round(float(x) / base))


class FilterScheduler(driver.Scheduler):
    """Scheduler that can be used for filtering and weighing."""
    def __init__(self, *args, **kwargs):
        super(FilterScheduler, self).__init__(*args, **kwargs)
        self.notifier = rpc.get_notifier('scheduler')
        scheduler_client = client.SchedulerClient()
        self.placement_client = scheduler_client.reportclient
        # WRS - network provider filter
        self.network_api = network.API()

    def run_periodic_tasks(self, context):
        """Called from a periodic tasks in the manager."""
        # WRS: Fetch hosts that are to be given preferential weight. Obtain
        # information in advance so that user requests do not have to wait.
        elevated = context.elevated()

        # Hosts that are patch current while some hosts are not yet current.
        try:
            with eventlet.Timeout(_query_timeout_seconds):
                patched_hosts = sorted(list(
                    self.host_manager.get_all_wrs_host_patch_prefer(elevated)))
        except eventlet.Timeout as e:
            patched_hosts = []
            LOG.error('get_all_wrs_host_patch_prefer() Timeout: %s', e)

        # Hosts that are upgrades current while in progress of upgrades.
        try:
            with eventlet.Timeout(_query_timeout_seconds):
                upgraded_hosts = sorted(list(
                    self.host_manager.get_all_wrs_host_upgrade_prefer(
                        elevated)))
        except eventlet.Timeout as e:
            upgraded_hosts = []
            LOG.error('get_all_wrs_host_upgrade_prefer() Timeout: %s', e)

        if patched_hosts or upgraded_hosts:
            LOG.info(
                'software current preferred hosts: '
                'patched=(%(P)s); upgraded=(%(U)s)',
                {'P': ', '.join(str(e) for e in patched_hosts),
                 'U': ', '.join(str(e) for e in upgraded_hosts)})

    def select_destinations(self, context, spec_obj, instance_uuids,
            alloc_reqs_by_rp_uuid, provider_summaries):
        """Returns a sorted list of HostState objects that satisfy the
        supplied request_spec.

        These hosts will have already had their resources claimed in Placement.

        :param context: The RequestContext object
        :param spec_obj: The RequestSpec object
        :param instance_uuids: List of UUIDs, one for each value of the spec
                               object's num_instances attribute
        :param alloc_reqs_by_rp_uuid: Optional dict, keyed by resource provider
                                      UUID, of the allocation requests that may
                                      be used to claim resources against
                                      matched hosts. If None, indicates either
                                      the placement API wasn't reachable or
                                      that there were no allocation requests
                                      returned by the placement API. If the
                                      latter, the provider_summaries will be an
                                      empty dict, not None.
        :param provider_summaries: Optional dict, keyed by resource provider
                                   UUID, of information that will be used by
                                   the filters/weighers in selecting matching
                                   hosts for a request. If None, indicates that
                                   the scheduler driver should grab all compute
                                   node information locally and that the
                                   Placement API is not used. If an empty dict,
                                   indicates the Placement API returned no
                                   potential matches for the requested
                                   resources.
        """
        self.notifier.info(
            context, 'scheduler.select_destinations.start',
            dict(request_spec=spec_obj.to_legacy_request_spec_dict()))

        # NOTE(sbauza): The RequestSpec.num_instances field contains the number
        # of instances created when the RequestSpec was used to first boot some
        # instances. This is incorrect when doing a move or resize operation,
        # so prefer the length of instance_uuids unless it is None.
        num_instances = (len(instance_uuids) if instance_uuids
                         else spec_obj.num_instances)

        # WRS: check against minimum number of instances for success if set
        #      otherwise default to num_instances
        if hasattr(spec_obj, 'min_num_instances'):
            task_state = spec_obj.scheduler_hints.get('task_state')
            # task_state set to None indicates this is not for migration
            if task_state is None:
                num_instances = spec_obj.min_num_instances

        selected_hosts = self._schedule(context, spec_obj, instance_uuids,
            alloc_reqs_by_rp_uuid, provider_summaries)

        # Couldn't fulfill the request_spec
        if len(selected_hosts) < num_instances:
            # NOTE(Rui Chen): If multiple creates failed, set the updated time
            # of selected HostState to None so that these HostStates are
            # refreshed according to database in next schedule, and release
            # the resource consumed by instance in the process of selecting
            # host.
            for host in selected_hosts:
                host.updated = None

            # Log the details but don't put those into the reason since
            # we don't want to give away too much information about our
            # actual environment.
            LOG.debug('There are %(hosts)d hosts available but '
                      '%(num_instances)d instances requested to build.',
                      {'hosts': len(selected_hosts),
                       'num_instances': num_instances})

            # Determine normalized resource allocation request required to do
            # placement query.
            resources = scheduler_utils.resources_from_request_spec(spec_obj)
            empty_computenode = objects.ComputeNode(
                numa_topology=objects.NUMATopology(
                    cells=[objects.NUMACell(siblings=[])])._to_json())
            normalized_resources = \
                scheduler_utils.normalized_resources_for_placement_claim(
                    resources, empty_computenode,
                    spec_obj.flavor.vcpus,
                    spec_obj.flavor.extra_specs,
                    spec_obj.image.properties,
                    spec_obj.numa_topology)

            # Determine the rejection reasons for all hosts based on
            # placement vcpu, memory, and disk criteria. This is done
            # after-the-fact since the placement query does not return
            # any reasons.
            reasons = self.placement_client.get_rejection_reasons(
                requested=normalized_resources)
            if reasons is None:
                reasons = {}

            # Populate per-host rejection map based on placement criteria.
            host_states = self.host_manager.get_all_host_states(context)
            for host_state in host_states:
                if host_state.uuid in reasons:
                    msg = reasons[host_state.uuid]
                    if msg:
                        utils.filter_reject('Placement',
                                                 host_state, spec_obj, msg,
                                                 append=False)

            # WRS - failure message
            pp = pprint.PrettyPrinter(indent=1)
            spec_ = {k.lstrip('_obj_'): v for k, v in
                     (spec_obj.__dict__).items() if k.startswith('_obj_')}
            LOG.warning('CANNOT SCHEDULE:  %(num)s available out of '
                        '%(req)s requested.  spec_obj=\n%(spec)s',
                        {'num': len(selected_hosts),
                         'req': num_instances,
                         'spec': pp.pformat(spec_),
                         })
            reason = _('There are not enough hosts available.')
            filter_properties = spec_obj.to_legacy_filter_properties_dict()
            scheduler_utils.NoValidHost_extend(filter_properties,
                                               reason=reason)
        else:
            # WRS - success message
            LOG.info(
                'SCHED: PASS. Selected %(hosts)s, uuid=%(uuid)s, '
                'name=%(name)s, display_name=%(display_name)s, '
                'scheduled=%(num)s',
                {'hosts': selected_hosts, 'uuid': spec_obj.instance_uuid,
                 'name': spec_obj.name, 'display_name': spec_obj.display_name,
                 'num': len(selected_hosts)})

        self.notifier.info(
            context, 'scheduler.select_destinations.end',
            dict(request_spec=spec_obj.to_legacy_request_spec_dict()))
        return selected_hosts

    def _schedule(self, context, spec_obj, instance_uuids,
            alloc_reqs_by_rp_uuid, provider_summaries):
        """Returns a list of hosts that meet the required specs, ordered by
        their fitness.

        These hosts will have already had their resources claimed in Placement.

        :param context: The RequestContext object
        :param spec_obj: The RequestSpec object
        :param instance_uuids: List of instance UUIDs to place or move.
        :param alloc_reqs_by_rp_uuid: Optional dict, keyed by resource provider
                                      UUID, of the allocation requests that may
                                      be used to claim resources against
                                      matched hosts. If None, indicates either
                                      the placement API wasn't reachable or
                                      that there were no allocation requests
                                      returned by the placement API. If the
                                      latter, the provider_summaries will be an
                                      empty dict, not None.
        :param provider_summaries: Optional dict, keyed by resource provider
                                   UUID, of information that will be used by
                                   the filters/weighers in selecting matching
                                   hosts for a request. If None, indicates that
                                   the scheduler driver should grab all compute
                                   node information locally and that the
                                   Placement API is not used. If an empty dict,
                                   indicates the Placement API returned no
                                   potential matches for the requested
                                   resources.
        """
        elevated = context.elevated()
        admin_context = novacontext.get_admin_context()

        # WRS - network provider filter
        physkey = 'provider:physical_network'
        scheduler_hints = {}
        if spec_obj.obj_attr_is_set('scheduler_hints'):
            scheduler_hints = spec_obj.scheduler_hints or {}
        if not scheduler_hints.get(physkey):
            hint = {physkey: set()}
            try:
                instance = objects.BuildRequest.get_by_instance_uuid(
                    context, spec_obj.instance_uuid).instance
            except exception.BuildRequestNotFound as err:
                # This covers for the reschedule case, at which point
                # the build request object is deleted, and
                # the instance object now exists in the database
                instance = objects.Instance.get_by_uuid(context,
                                                    spec_obj.instance_uuid)
            network_info = instance.info_cache.network_info
            for ntwk in network_info:
                network_uuid = ntwk['network']['id']
                vif_model = ntwk['vif_model']
                try:
                    # Set provider network scheduler hint
                    # only if a virtual interface
                    if not pci_utils.vif_model_pci_passthrough(vif_model):
                        net_info = self.network_api.get_dict(admin_context,
                                                             network_uuid)
                        physnet = net_info.get(physkey, None)
                        if physnet is not None:
                            hint[physkey].update([physnet])
                except Exception as err:
                    LOG.warning("Cannot get %(physkey)r "
                                    "from network=%(network_uuid)r, "
                                    "error=%(err)r",
                                {'physkey': physkey,
                                 'network_uuid': network_uuid,
                                 'err': err})
            if len(hint[physkey]) > 0:
                scheduler_hints.update({physkey: list(hint[physkey])})
                spec_obj.scheduler_hints = scheduler_hints

        # Find our local list of acceptable hosts by repeatedly
        # filtering and weighing our options. Each time we choose a
        # host, we virtually consume resources on it so subsequent
        # selections can adjust accordingly.

        # Note: remember, we are using a generator-iterator here. So only
        # traverse this list once. This can bite you if the hosts
        # are being scanned in a filter or weighing function.
        hosts = self._get_all_host_states(elevated, spec_obj,
            provider_summaries)

        # A list of the instance UUIDs that were successfully claimed against
        # in the placement API. If we are not able to successfully claim for
        # all involved instances, we use this list to remove those allocations
        # before returning
        claimed_instance_uuids = []

        # WRS:extension -- group policy best effort option
        # Get metadata for further checks
        md = {}
        if spec_obj.instance_group:
            md = spec_obj.instance_group.get('metadetails', {})

        group_best_effort = strutils.bool_from_string(
            md.get('wrs-sg:best_effort', False))
        # If using best-effort, we need to keep hosts as a list
        # so we can revert to the saved version
        if group_best_effort:
            hosts = list(hosts)
            orig_hosts = hosts

        selected_hosts = []

        # NOTE(sbauza): The RequestSpec.num_instances field contains the number
        # of instances created when the RequestSpec was used to first boot some
        # instances. This is incorrect when doing a move or resize operation,
        # so prefer the length of instance_uuids unless it is None.
        num_instances = (len(instance_uuids) if instance_uuids
                         else spec_obj.num_instances)
        for num in range(num_instances):
            hosts = self._get_sorted_hosts(spec_obj, hosts, num)

            # WRS:extension - for group affinity with best effort enforcement,
            # if we cannot find hosts meeting affinity policy, then relax the
            # restriction and try again.
            if (not hosts and group_best_effort and
                    spec_obj.instance_group.hosts):
                LOG.info('Group affinity: no hosts meeting affinity '
                         'policy, retrying using best-effort.')
                spec_obj.instance_group.hosts = []
                # hosts has to be not part of the updates when saving
                spec_obj.instance_group.obj_reset_changes(['hosts'])
                # set index to 0 to make sure all filters are used
                hosts = self._get_sorted_hosts(spec_obj, orig_hosts, 0)

            if not hosts:
                # NOTE(jaypipes): If we get here, that means not all instances
                # in instance_uuids were able to be matched to a selected host.
                # So, let's clean up any already-claimed allocations here
                # before breaking and returning
                self._cleanup_allocations(claimed_instance_uuids)
                break

            if (instance_uuids is None or
                    not self.USES_ALLOCATION_CANDIDATES or
                    alloc_reqs_by_rp_uuid is None):
                # Unfortunately, we still need to deal with older conductors
                # that may not be passing in a list of instance_uuids. In those
                # cases, obviously we can't claim resources because we don't
                # have instance UUIDs to claim with, so we just grab the first
                # host in the list of sorted hosts. In addition to older
                # conductors, we need to support the caching scheduler, which
                # doesn't use the placement API (and has
                # USES_ALLOCATION_CANDIDATE = False) and therefore we skip all
                # the claiming logic for that scheduler driver. Finally, if
                # there was a problem communicating with the placement API,
                # alloc_reqs_by_rp_uuid will be None, so we skip claiming in
                # that case as well
                claimed_host = hosts[0]
            else:
                instance_uuid = instance_uuids[num]

                # Attempt to claim the resources against one or more resource
                # providers, looping over the sorted list of possible hosts
                # looking for an allocation request that contains that host's
                # resource provider UUID
                claimed_host = None
                for host in hosts:
                    cn_uuid = host.uuid
                    if cn_uuid not in alloc_reqs_by_rp_uuid:
                        LOG.debug("Found host state %s that wasn't in "
                                  "allocation requests. Skipping.", cn_uuid)
                        continue

                    alloc_reqs = alloc_reqs_by_rp_uuid[cn_uuid]
                    # WRS: Determine resources consumed and update placement
                    # service
                    for alloc in alloc_reqs[0]['allocations']:
                        if alloc['resource_provider']['uuid'] == cn_uuid:
                            vcpus = (spec_obj.flavor.vcpus -
                                               spec_obj.offline_cpus)
                            extra_specs = spec_obj.flavor.extra_specs
                            image_props = spec_obj.image.properties
                            instance_numa_topology = spec_obj.numa_topology
                            resources = alloc['resources']
                            normalized_resources = scheduler_utils. \
                                normalized_resources_for_placement_claim(
                                    resources, host, vcpus, extra_specs,
                                    image_props, instance_numa_topology)
                            alloc['resources'] = normalized_resources

                    if self._claim_resources(elevated, spec_obj, instance_uuid,
                            alloc_reqs):
                        claimed_host = host
                        break

                if claimed_host is None:
                    # We weren't able to claim resources in the placement API
                    # for any of the sorted hosts identified. So, clean up any
                    # successfully-claimed resources for prior instances in
                    # this request and return an empty list which will cause
                    # select_destinations() to raise NoValidHost
                    LOG.debug("Unable to successfully claim against any host.")
                    self._cleanup_allocations(claimed_instance_uuids)
                    return []

                claimed_instance_uuids.append(instance_uuid)

            LOG.debug("Selected host: %(host)s", {'host': claimed_host})
            selected_hosts.append(claimed_host)

            # Now consume the resources so the filter/weights will change for
            # the next instance.
            claimed_host.consume_from_request(spec_obj)
            if spec_obj.instance_group is not None:
                spec_obj.instance_group.hosts.append(claimed_host.host)
                # hosts has to be not part of the updates when saving
                spec_obj.instance_group.obj_reset_changes(['hosts'])
        return selected_hosts

    def _cleanup_allocations(self, instance_uuids):
        """Removes allocations for the supplied instance UUIDs."""
        if not instance_uuids:
            return
        LOG.debug("Cleaning up allocations for %s", instance_uuids)
        for uuid in instance_uuids:
            self.placement_client.delete_allocation_for_instance(uuid)

    def _claim_resources(self, ctx, spec_obj, instance_uuid, alloc_reqs):
        """Given an instance UUID (representing the consumer of resources), the
        HostState object for the host that was chosen for the instance, and a
        list of allocation request JSON objects, attempt to claim resources for
        the instance in the placement API. Returns True if the claim process
        was successful, False otherwise.

        :param ctx: The RequestContext object
        :param spec_obj: The RequestSpec object
        :param instance_uuid: The UUID of the consuming instance
        :param cn_uuid: UUID of the host to allocate against
        :param alloc_reqs: A list of allocation request JSON objects that
                           allocate against (at least) the compute host
                           selected by the _schedule() method. These allocation
                           requests were constructed from a call to the GET
                           /allocation_candidates placement API call.  Each
                           allocation_request satisfies the original request
                           for resources and can be supplied as-is (along with
                           the project and user ID to the placement API's
                           PUT /allocations/{consumer_uuid} call to claim
                           resources for the instance
        """

        if scheduler_utils.request_is_rebuild(spec_obj):
            # NOTE(danms): This is a rebuild-only scheduling request, so we
            # should not be doing any extra claiming
            LOG.debug('Not claiming resources in the placement API for '
                      'rebuild-only scheduling of instance %(uuid)s',
                      {'uuid': instance_uuid})
            return True

        LOG.debug("Attempting to claim resources in the placement API for "
                  "instance %s", instance_uuid)

        project_id = spec_obj.project_id

        # NOTE(jaypipes): So, the RequestSpec doesn't store the user_id,
        # only the project_id, so we need to grab the user information from
        # the context. Perhaps we should consider putting the user ID in
        # the spec object?
        user_id = ctx.user_id

        # TODO(jaypipes): Loop through all allocation requests instead of just
        # trying the first one. For now, since we'll likely want to order the
        # allocation requests in the future based on information in the
        # provider summaries, we'll just try to claim resources using the first
        # allocation request
        alloc_req = alloc_reqs[0]

        return self.placement_client.claim_resources(instance_uuid,
            alloc_req, project_id, user_id)

    def _get_sorted_hosts(self, spec_obj, host_states, index):
        """Returns a list of HostState objects that match the required
        scheduling constraints for the request spec object and have been sorted
        according to the weighers.
        """
        filtered_hosts = self.host_manager.get_filtered_hosts(host_states,
            spec_obj, index)

        LOG.debug("Filtered %(hosts)s", {'hosts': filtered_hosts})

        if not filtered_hosts:
            return []

        weighed_hosts_cls = self.host_manager.get_weighed_hosts(filtered_hosts,
            spec_obj)
        # Strip off the WeighedHost wrapper class...
        weighed_hosts = [h.obj for h in weighed_hosts_cls]

        LOG.debug("Weighed %(hosts)s", {'hosts': weighed_hosts})

        # WRS: Enable grouping of hosts of similar weight versus using
        # subset_size option.
        chosen_host = None
        if CONF.filter_scheduler.rounded_weight > 0:
            # Group the weighed hosts into subsets of similar weight,
            # then randomly select a host from the first group.
            # This assumes list was pre-sorted.
            grouped_list = \
                [(mround(x.weight,
                         base=CONF.filter_scheduler.rounded_weight), x)
                 for x in weighed_hosts_cls]
            for key, group in itertools.groupby(grouped_list,
                                                lambda x: x[0]):
                chosen_host_cls = random.choice(list(group))[1]
                chosen_host = chosen_host_cls.obj
                break
            # Display grouped weigher details
            for key, group in itertools.groupby(grouped_list, lambda x: x[0]):
                G = [g[1] for g in list(group)]
                LOG.info('GROUP: {}, {}'.format(key, G))
        else:
            # We randomize the first element in the returned list to alleviate
            # congestion where the same host is consistently selected among
            # numerous potential hosts for similar request specs.
            host_subset_size = CONF.filter_scheduler.host_subset_size
            if host_subset_size < len(weighed_hosts):
                weighed_subset = weighed_hosts[0:host_subset_size]
            else:
                weighed_subset = weighed_hosts
            chosen_host = random.choice(weighed_subset)

        weighed_hosts.remove(chosen_host)
        return [chosen_host] + weighed_hosts

    def _get_all_host_states(self, context, spec_obj, provider_summaries):
        """Template method, so a subclass can implement caching."""
        # NOTE(jaypipes): provider_summaries being None is treated differently
        # from an empty dict. provider_summaries is None when we want to grab
        # all compute nodes, for instance when using the caching scheduler.
        # The provider_summaries variable will be an empty dict when the
        # Placement API found no providers that match the requested
        # constraints, which in turn makes compute_uuids an empty list and
        # get_host_states_by_uuids will return an empty generator-iterator
        # also, which will eventually result in a NoValidHost error.
        compute_uuids = None
        if provider_summaries is not None:
            compute_uuids = list(provider_summaries.keys())
        return self.host_manager.get_host_states_by_uuids(context,
                                                          compute_uuids,
                                                          spec_obj)
