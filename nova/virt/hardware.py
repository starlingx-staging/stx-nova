# Copyright 2014 Red Hat, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2013-2017 Wind River Systems, Inc.
#

import collections
import fractions
import itertools
import os
import sys

from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import units
import six

import nova.conf
from nova import context
from nova import exception
from nova.i18n import _
from nova import objects
from nova.objects import fields
from nova.objects import instance as obj_instance
from nova import utils

CONF = nova.conf.CONF
LOG = logging.getLogger(__name__)

MEMPAGES_SMALL = -1
MEMPAGES_LARGE = -2
MEMPAGES_ANY = -3

# WRS base path used for floating instance cpusets
CPUSET_BASE = '/sys/fs/cgroup/cpuset/floating'


# WRS - extra_specs, image_props helper functions
# NOTE: This must be consistent with _add_pinning_constraint().
def is_cpu_policy_dedicated(extra_specs, image_props):
    flavor_policy = extra_specs.get('hw:cpu_policy')
    image_policy = image_props.get('hw_cpu_policy')
    if (flavor_policy == fields.CPUAllocationPolicy.DEDICATED or
        image_policy == fields.CPUAllocationPolicy.DEDICATED):
        return True
    return False


# NOTE: This must be consistent with _add_pinning_constraint().
def is_cpu_thread_policy_isolate(extra_specs, image_props):
    flavor_thread_policy = extra_specs.get('hw:cpu_thread_policy')
    image_thread_policy = image_props.get('hw_cpu_thread_policy')
    if (flavor_thread_policy == fields.CPUThreadAllocationPolicy.ISOLATE or
        image_thread_policy == fields.CPUThreadAllocationPolicy.ISOLATE):
        return True
    return False


def _get_threads_per_core(host_numa_topology):
    """Get number of hyperthreading threads per core based on host numa
       topology siblings.
    """
    threads_per_core = 1
    if (host_numa_topology is None or
            not hasattr(host_numa_topology, 'cells')):
        return threads_per_core

    host_cell = host_numa_topology.cells[0]
    if host_cell.siblings:
        threads_per_core = max(map(len, host_cell.siblings))
    return threads_per_core


# WRS:extension - normalized vcpu accounting
# NOTE: We do not need to also check cpu_policy since flavors are validated.
def unshared_vcpus(vcpus, extra_specs):
    """Count of unshared vCPUs"""
    shared_vcpu = extra_specs.get('hw:wrs:shared_vcpu', None)
    if shared_vcpu is not None:
        vcpus -= 1
    return vcpus


def normalized_vcpus(vcpus=None, reserved=None, extra_specs=None,
                     image_props=None, ratio=None, threads_per_core=1):
    """Normalize vcpu used count based on cpu_policy, cpu_thread_policy,
       reserved_vcpus, and shared_vcpu.

       Accounting of vcpus_used is adjusted to be out of vcpus_total
       (i.e. pcpus) using fractional units. When vcpus_used reaches pcpus,
       the compute node is full.  This routine provides consistency across
       scheduler (i.e. core_filter, host_manager), and resource tracker..

       Pinned instances effectively have ratio = 1.0.
       Pinned instances with shared_vcpu require exclusion of 1 vcpu.
       Non-pinned instances (floaters) are divided by ratio.
    """
    if vcpus is None or ratio is None:
        return 0
    norm_vcpus = float(vcpus)
    if extra_specs is None:
        extra_specs = {}
    if image_props is None:
        image_props = {}
    is_dedicated = is_cpu_policy_dedicated(extra_specs, image_props)
    is_isolate = is_cpu_thread_policy_isolate(extra_specs, image_props)
    shared_vcpu = extra_specs.get('hw:wrs:shared_vcpu', None)
    if is_dedicated:
        if is_isolate and reserved is not None:
            n_reserved = len(reserved)
            if n_reserved > 0:
                norm_vcpus += len(reserved)
            else:
                norm_vcpus *= threads_per_core
        if shared_vcpu is not None:
            # NOTE(jgauld): should extend this to handle thread siblings
            norm_vcpus -= 1
    else:
        norm_vcpus = norm_vcpus / ratio

    # Get parent calling functions
    parent_fname1 = sys._getframe(1).f_code.co_name
    parent_fname2 = sys._getframe(2).f_code.co_name
    LOG.debug('{}.{}.normalized_vcpus(), '
              'vcpus={}, reserved={}, extra_specs={}, image_props={}, '
              'is_dedicated={}, is_isolate={}, shared_vcpu={}, '
              'threads_per_core={}, norm_vcpus={}'.
              format(parent_fname2, parent_fname1,
                     vcpus, reserved, extra_specs, image_props,
                     is_dedicated, is_isolate, shared_vcpu, threads_per_core,
                     norm_vcpus))
    return norm_vcpus


def get_reserved_thread_sibling_pcpus(instance_numa_topology=None,
                                      host_numa_topology=None):
    """Get set of reserved thread sibling pcpus.

    When using cpu_thread_policy = isolate, there are thread-siblings
    corresponding to cpu_pinned that cannot be used, so in effect they are
    reserved.  These are currently not stored in numa topology object.

    Returns a set of reserved thread sibling pcpus excluding self.
    """
    reserved = set()

    # Get parent calling functions
    parent_fname1 = sys._getframe(1).f_code.co_name
    parent_fname2 = sys._getframe(2).f_code.co_name

    if (instance_numa_topology is None or
            not hasattr(instance_numa_topology, 'cells')):
        LOG.warning(
            "%(name2)s.%(name1)s.get_reserved_thread_sibling_pcpus(), "
            "instance_numa_topology not present.",
            {'name1': parent_fname1, 'name2': parent_fname2})
        return reserved
    if (host_numa_topology is None or
            not hasattr(host_numa_topology, 'cells')):
        LOG.warning(
            "%(name2)s.%(name1)s.get_reserved_thread_sibling_pcpus(), "
            "host_numa_topology not present.",
            {'name1': parent_fname1, 'name2': parent_fname2})
        return reserved

    # Deduce the set of thread siblings for each sibling (excluding self),
    # based on host numa topology.
    siblings = {}
    for cell in host_numa_topology.cells:
        for sibs in cell.siblings:
            for e in sibs:
                siblings[e] = sibs.copy() - set([e])

    # Deduce set of pinned pcpus and reserved thread sibling pcpus from
    # instance numa topology.  Only need to do this for ISOLATE policy.
    cpu_thread_policy = instance_numa_topology.cells[0]['cpu_thread_policy']
    if cpu_thread_policy != fields.CPUThreadAllocationPolicy.ISOLATE:
        return reserved
    pinned = set()
    for cell in instance_numa_topology.cells:
        cpu_pinning = cell['cpu_pinning']
        if cpu_pinning is not None:
            for e in cpu_pinning.values():
                pinned.add(e)
                if e in siblings:
                    reserved.update(siblings[e])
    reserved.difference_update(pinned)
    return reserved


# WRS: shared pcpu extension
def get_shared_pcpu_map():
    """Parsing shared_pcpu_map config.

    Returns a map of numa nodes to shared pcpu indices
    """
    if not CONF.shared_pcpu_map:
        return {}
    shared_pcpu_map = CONF.shared_pcpu_map
    # Clean out invalid entries
    for k, v in shared_pcpu_map.items():
        if not v.isdigit():
            del shared_pcpu_map[k]
    return shared_pcpu_map


def get_vcpu_pin_set():
    """Parse vcpu_pin_set config.

    :returns: a set of pcpu ids can be used by instances
    """
    if not CONF.vcpu_pin_set:
        return None

    cpuset_ids = parse_cpu_spec(CONF.vcpu_pin_set)
    if not cpuset_ids:
        raise exception.Invalid(_("No CPUs available after parsing %r") %
                                CONF.vcpu_pin_set)
    return cpuset_ids


def parse_cpu_spec(spec):
    """Parse a CPU set specification.

    Each element in the list is either a single CPU number, a range of
    CPU numbers, or a caret followed by a CPU number to be excluded
    from a previous range.

    :param spec: cpu set string eg "1-4,^3,6"

    :returns: a set of CPU indexes
    """
    cpuset_ids = set()
    cpuset_reject_ids = set()
    for rule in spec.split(','):
        rule = rule.strip()
        # Handle multi ','
        if len(rule) < 1:
            continue
        # Note the count limit in the .split() call
        range_parts = rule.split('-', 1)
        if len(range_parts) > 1:
            reject = False
            if range_parts[0] and range_parts[0][0] == '^':
                reject = True
                range_parts[0] = str(range_parts[0][1:])

            # So, this was a range; start by converting the parts to ints
            try:
                start, end = [int(p.strip()) for p in range_parts]
            except ValueError:
                raise exception.Invalid(_("Invalid range expression %r")
                                        % rule)
            # Make sure it's a valid range
            if start > end:
                raise exception.Invalid(_("Invalid range expression %r")
                                        % rule)
            # Add available CPU ids to set
            if not reject:
                cpuset_ids |= set(range(start, end + 1))
            else:
                cpuset_reject_ids |= set(range(start, end + 1))
        elif rule[0] == '^':
            # Not a range, the rule is an exclusion rule; convert to int
            try:
                cpuset_reject_ids.add(int(rule[1:].strip()))
            except ValueError:
                raise exception.Invalid(_("Invalid exclusion "
                                          "expression %r") % rule)
        else:
            # OK, a single CPU to include; convert to int
            try:
                cpuset_ids.add(int(rule))
            except ValueError:
                raise exception.Invalid(_("Invalid inclusion "
                                          "expression %r") % rule)

    # Use sets to handle the exclusion rules for us
    cpuset_ids -= cpuset_reject_ids

    return cpuset_ids


def format_cpu_spec(cpuset, allow_ranges=True):
    """Format a libvirt CPU range specification.

    Format a set/list of CPU indexes as a libvirt CPU range
    specification. It allow_ranges is true, it will try to detect
    continuous ranges of CPUs, otherwise it will just list each CPU
    index explicitly.

    :param cpuset: set (or list) of CPU indexes

    :returns: a formatted CPU range string
    """
    # We attempt to detect ranges, but don't bother with
    # trying to do range negations to minimize the overall
    # spec string length
    if allow_ranges:
        ranges = []
        previndex = None
        for cpuindex in sorted(cpuset):
            if previndex is None or previndex != (cpuindex - 1):
                ranges.append([])
            ranges[-1].append(cpuindex)
            previndex = cpuindex

        parts = []
        for entry in ranges:
            if len(entry) == 1:
                parts.append(str(entry[0]))
            else:
                parts.append("%d-%d" % (entry[0], entry[len(entry) - 1]))
        return ",".join(parts)
    else:
        return ",".join(str(id) for id in sorted(cpuset))


def get_number_of_serial_ports(flavor, image_meta):
    """Get the number of serial consoles from the flavor or image.

    If flavor extra specs is not set, then any image meta value is
    permitted.  If flavor extra specs *is* set, then this provides the
    default serial port count. The image meta is permitted to override
    the extra specs, but *only* with a lower value, i.e.:

    - flavor hw:serial_port_count=4
      VM gets 4 serial ports
    - flavor hw:serial_port_count=4 and image hw_serial_port_count=2
      VM gets 2 serial ports
    - image hw_serial_port_count=6
      VM gets 6 serial ports
    - flavor hw:serial_port_count=4 and image hw_serial_port_count=6
      Abort guest boot - forbidden to exceed flavor value

    :param flavor: Flavor object to read extra specs from
    :param image_meta: nova.objects.ImageMeta object instance

    :returns: number of serial ports
    """

    def get_number(obj, property):
        num_ports = obj.get(property)
        if num_ports is not None:
            try:
                num_ports = int(num_ports)
            except ValueError:
                raise exception.ImageSerialPortNumberInvalid(
                    num_ports=num_ports, property=property)
        return num_ports

    flavor_num_ports = get_number(flavor.extra_specs, "hw:serial_port_count")
    image_num_ports = image_meta.properties.get("hw_serial_port_count", None)

    if (flavor_num_ports and image_num_ports) is not None:
        if image_num_ports > flavor_num_ports:
            raise exception.ImageSerialPortNumberExceedFlavorValue()
        return image_num_ports

    return flavor_num_ports or image_num_ports or 1


class InstanceInfo(object):

    def __init__(self, state=None, max_mem_kb=0, mem_kb=0, num_cpu=0,
                 cpu_time_ns=0, id=None):
        """Create a new Instance Info object

        :param state: the running state, one of the power_state codes
        :param max_mem_kb: (int) the maximum memory in KBytes allowed
        :param mem_kb: (int) the memory in KBytes used by the instance
        :param num_cpu: (int) the number of virtual CPUs for the
                        instance
        :param cpu_time_ns: (int) the CPU time used in nanoseconds
        :param id: a unique ID for the instance
        """
        self.state = state
        self.max_mem_kb = max_mem_kb
        self.mem_kb = mem_kb
        self.num_cpu = num_cpu
        self.cpu_time_ns = cpu_time_ns
        self.id = id

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.__dict__ == other.__dict__)


def _score_cpu_topology(topology, wanttopology):
    """Compare a topology against a desired configuration.

    Calculate a score indicating how well a provided topology matches
    against a preferred topology, where:

     a score of 3 indicates an exact match for sockets, cores and
       threads
     a score of 2 indicates a match of sockets and cores, or sockets
       and threads, or cores and threads
     a score of 1 indicates a match of sockets or cores or threads
     a score of 0 indicates no match

    :param wanttopology: nova.objects.VirtCPUTopology instance for
                         preferred topology

    :returns: score in range 0 (worst) to 3 (best)
    """
    score = 0
    if (wanttopology.sockets != -1 and
        topology.sockets == wanttopology.sockets):
        score = score + 1
    if (wanttopology.cores != -1 and
        topology.cores == wanttopology.cores):
        score = score + 1
    if (wanttopology.threads != -1 and
        topology.threads == wanttopology.threads):
        score = score + 1
    return score


def _get_cpu_topology_constraints(flavor, image_meta):
    """Get the topology constraints declared in flavor or image

    Extracts the topology constraints from the configuration defined in
    the flavor extra specs or the image metadata. In the flavor this
    will look for:

     hw:cpu_sockets - preferred socket count
     hw:cpu_cores - preferred core count
     hw:cpu_threads - preferred thread count
     hw:cpu_max_sockets - maximum socket count
     hw:cpu_max_cores - maximum core count
     hw:cpu_max_threads - maximum thread count

    In the image metadata this will look at:

     hw_cpu_sockets - preferred socket count
     hw_cpu_cores - preferred core count
     hw_cpu_threads - preferred thread count
     hw_cpu_max_sockets - maximum socket count
     hw_cpu_max_cores - maximum core count
     hw_cpu_max_threads - maximum thread count

    The image metadata must be strictly lower than any values set in
    the flavor. All values are, however, optional.

    :param flavor: Flavor object to read extra specs from
    :param image_meta: nova.objects.ImageMeta object instance

    :raises: exception.ImageVCPULimitsRangeExceeded if the maximum
             counts set against the image exceed the maximum counts
             set against the flavor
    :raises: exception.ImageVCPUTopologyRangeExceeded if the preferred
             counts set against the image exceed the maximum counts set
             against the image or flavor
    :returns: A two-tuple of objects.VirtCPUTopology instances. The
              first element corresponds to the preferred topology,
              while the latter corresponds to the maximum topology,
              based on upper limits.
    """
    # Obtain the absolute limits from the flavor
    flvmaxsockets = int(flavor.extra_specs.get(
        "hw:cpu_max_sockets", 65536))
    flvmaxcores = int(flavor.extra_specs.get(
        "hw:cpu_max_cores", 65536))
    flvmaxthreads = int(flavor.extra_specs.get(
        "hw:cpu_max_threads", 65536))

    LOG.debug("Flavor limits %(sockets)d:%(cores)d:%(threads)d",
              {"sockets": flvmaxsockets,
               "cores": flvmaxcores,
               "threads": flvmaxthreads})

    # Get any customized limits from the image
    props = image_meta.properties
    maxsockets = props.get("hw_cpu_max_sockets", flvmaxsockets)
    maxcores = props.get("hw_cpu_max_cores", flvmaxcores)
    maxthreads = props.get("hw_cpu_max_threads", flvmaxthreads)

    LOG.debug("Image limits %(sockets)d:%(cores)d:%(threads)d",
              {"sockets": maxsockets,
               "cores": maxcores,
               "threads": maxthreads})

    # Image limits are not permitted to exceed the flavor
    # limits. ie they can only lower what the flavor defines
    if ((maxsockets > flvmaxsockets) or
        (maxcores > flvmaxcores) or
        (maxthreads > flvmaxthreads)):
        raise exception.ImageVCPULimitsRangeExceeded(
            sockets=maxsockets,
            cores=maxcores,
            threads=maxthreads,
            maxsockets=flvmaxsockets,
            maxcores=flvmaxcores,
            maxthreads=flvmaxthreads)

    # Get any default preferred topology from the flavor
    flvsockets = int(flavor.extra_specs.get("hw:cpu_sockets", -1))
    flvcores = int(flavor.extra_specs.get("hw:cpu_cores", -1))
    flvthreads = int(flavor.extra_specs.get("hw:cpu_threads", -1))

    LOG.debug("Flavor pref %(sockets)d:%(cores)d:%(threads)d",
              {"sockets": flvsockets,
               "cores": flvcores,
               "threads": flvthreads})

    # If the image limits have reduced the flavor limits
    # we might need to discard the preferred topology
    # from the flavor
    if ((flvsockets > maxsockets) or
        (flvcores > maxcores) or
        (flvthreads > maxthreads)):
        flvsockets = flvcores = flvthreads = -1

    # Finally see if the image has provided a preferred
    # topology to use
    sockets = props.get("hw_cpu_sockets", -1)
    cores = props.get("hw_cpu_cores", -1)
    threads = props.get("hw_cpu_threads", -1)

    LOG.debug("Image pref %(sockets)d:%(cores)d:%(threads)d",
              {"sockets": sockets,
               "cores": cores,
               "threads": threads})

    # Image topology is not permitted to exceed image/flavor
    # limits
    if ((sockets > maxsockets) or
        (cores > maxcores) or
        (threads > maxthreads)):
        raise exception.ImageVCPUTopologyRangeExceeded(
            sockets=sockets,
            cores=cores,
            threads=threads,
            maxsockets=maxsockets,
            maxcores=maxcores,
            maxthreads=maxthreads)

    # If no preferred topology was set against the image
    # then use the preferred topology from the flavor
    # We use 'and' not 'or', since if any value is set
    # against the image this invalidates the entire set
    # of values from the flavor
    if sockets == -1 and cores == -1 and threads == -1:
        sockets = flvsockets
        cores = flvcores
        threads = flvthreads

    LOG.debug("Chosen %(sockets)d:%(cores)d:%(threads)d limits "
              "%(maxsockets)d:%(maxcores)d:%(maxthreads)d",
              {"sockets": sockets, "cores": cores,
               "threads": threads, "maxsockets": maxsockets,
               "maxcores": maxcores, "maxthreads": maxthreads})

    return (objects.VirtCPUTopology(sockets=sockets, cores=cores,
                                    threads=threads),
            objects.VirtCPUTopology(sockets=maxsockets, cores=maxcores,
                                    threads=maxthreads))


def _get_possible_cpu_topologies(vcpus, maxtopology,
                                 allow_threads):
    """Get a list of possible topologies for a vCPU count.

    Given a total desired vCPU count and constraints on the maximum
    number of sockets, cores and threads, return a list of
    objects.VirtCPUTopology instances that represent every possible
    topology that satisfies the constraints.

    :param vcpus: total number of CPUs for guest instance
    :param maxtopology: objects.VirtCPUTopology instance for upper
                        limits
    :param allow_threads: True if the hypervisor supports CPU threads

    :raises: exception.ImageVCPULimitsRangeImpossible if it is
             impossible to achieve the total vcpu count given
             the maximum limits on sockets, cores and threads
    :returns: list of objects.VirtCPUTopology instances
    """
    # Clamp limits to number of vcpus to prevent
    # iterating over insanely large list
    maxsockets = min(vcpus, maxtopology.sockets)
    maxcores = min(vcpus, maxtopology.cores)
    maxthreads = min(vcpus, maxtopology.threads)

    if not allow_threads:
        maxthreads = 1

    LOG.debug("Build topologies for %(vcpus)d vcpu(s) "
              "%(maxsockets)d:%(maxcores)d:%(maxthreads)d",
              {"vcpus": vcpus, "maxsockets": maxsockets,
               "maxcores": maxcores, "maxthreads": maxthreads})

    # Figure out all possible topologies that match
    # the required vcpus count and satisfy the declared
    # limits. If the total vCPU count were very high
    # it might be more efficient to factorize the vcpu
    # count and then only iterate over its factors, but
    # that's overkill right now
    possible = []
    for s in range(1, maxsockets + 1):
        for c in range(1, maxcores + 1):
            for t in range(1, maxthreads + 1):
                if (t * c * s) != vcpus:
                    continue
                possible.append(
                    objects.VirtCPUTopology(sockets=s,
                                            cores=c,
                                            threads=t))

    # We want to
    #  - Minimize threads (ie larger sockets * cores is best)
    #  - Prefer sockets over cores
    possible = sorted(possible, reverse=True,
                      key=lambda x: (x.sockets * x.cores,
                                     x.sockets,
                                     x.threads))

    LOG.debug("Got %d possible topologies", len(possible))
    if len(possible) == 0:
        raise exception.ImageVCPULimitsRangeImpossible(vcpus=vcpus,
                                                       sockets=maxsockets,
                                                       cores=maxcores,
                                                       threads=maxthreads)

    return possible


def _filter_for_numa_threads(possible, wantthreads):
    """Filter topologies which closest match to NUMA threads.

    Determine which topologies provide the closest match to the number
    of threads desired by the NUMA topology of the instance.

    The possible topologies may not have any entries which match the
    desired thread count. This method will find the topologies which
    have the closest matching count. For example, if 'wantthreads' is 4
    and the possible topologies has entries with 6, 3, 2 or 1 threads,
    the topologies which have 3 threads will be identified as the
    closest match not greater than 4 and will be returned.

    :param possible: list of objects.VirtCPUTopology instances
    :param wantthreads: desired number of threads

    :returns: list of objects.VirtCPUTopology instances
    """
    # First figure out the largest available thread
    # count which is not greater than wantthreads
    mostthreads = 0
    for topology in possible:
        if topology.threads > wantthreads:
            continue
        if topology.threads > mostthreads:
            mostthreads = topology.threads

    # Now restrict to just those topologies which
    # match the largest thread count
    bestthreads = []
    for topology in possible:
        if topology.threads != mostthreads:
            continue
        bestthreads.append(topology)

    return bestthreads


def _sort_possible_cpu_topologies(possible, wanttopology):
    """Sort the topologies in order of preference.

    Sort the provided list of possible topologies such that the
    configurations which most closely match the preferred topology are
    first.

    :param possible: list of objects.VirtCPUTopology instances
    :param wanttopology: objects.VirtCPUTopology instance for preferred
                         topology

    :returns: sorted list of nova.objects.VirtCPUTopology instances
    """

    # Look at possible topologies and score them according
    # to how well they match the preferred topologies
    # We don't use python's sort(), since we want to
    # preserve the sorting done when populating the
    # 'possible' list originally
    scores = collections.defaultdict(list)
    for topology in possible:
        score = _score_cpu_topology(topology, wanttopology)
        scores[score].append(topology)

    # Build list of all possible topologies sorted
    # by the match score, best match first
    desired = []
    desired.extend(scores[3])
    desired.extend(scores[2])
    desired.extend(scores[1])
    desired.extend(scores[0])

    return desired


def _get_desirable_cpu_topologies(flavor, image_meta, allow_threads=True,
                                  numa_topology=None):
    """Identify desirable CPU topologies based for given constraints.

    Look at the properties set in the flavor extra specs and the image
    metadata and build up a list of all possible valid CPU topologies
    that can be used in the guest. Then return this list sorted in
    order of preference.

    :param flavor: objects.Flavor instance to query extra specs from
    :param image_meta: nova.objects.ImageMeta object instance
    :param allow_threads: if the hypervisor supports CPU threads
    :param numa_topology: objects.InstanceNUMATopology instance that
                          may contain additional topology constraints
                          (such as threading information) that should
                          be considered

    :returns: sorted list of objects.VirtCPUTopology instances
    """

    LOG.debug("Getting desirable topologies for flavor %(flavor)s "
              "and image_meta %(image_meta)s, allow threads: %(threads)s",
              {"flavor": flavor, "image_meta": image_meta,
               "threads": allow_threads})

    preferred, maximum = _get_cpu_topology_constraints(flavor, image_meta)
    LOG.debug("Topology preferred %(preferred)s, maximum %(maximum)s",
              {"preferred": preferred, "maximum": maximum})

    possible = _get_possible_cpu_topologies(flavor.vcpus,
                                            maximum,
                                            allow_threads)
    LOG.debug("Possible topologies %s", possible)

    if numa_topology:
        min_requested_threads = None
        cell_topologies = [cell.cpu_topology for cell in numa_topology.cells
                           if ('cpu_topology' in cell
                               and cell.cpu_topology)]
        if cell_topologies:
            min_requested_threads = min(
                    topo.threads for topo in cell_topologies)

        if min_requested_threads:
            if preferred.threads != -1:
                min_requested_threads = min(preferred.threads,
                                            min_requested_threads)
            # WRS: If flavor is using shared_vcpu, add an extra thread
            shared_vcpu = flavor.get('extra_specs',
                                     {}).get("hw:wrs:shared_vcpu", None)
            if shared_vcpu is not None:
                min_requested_threads = min_requested_threads + 1
            specified_threads = max(1, min_requested_threads)
            LOG.debug("Filtering topologies best for %d threads",
                      specified_threads)

            possible = _filter_for_numa_threads(possible,
                                                specified_threads)
            LOG.debug("Remaining possible topologies %s",
                      possible)

    desired = _sort_possible_cpu_topologies(possible, preferred)
    LOG.debug("Sorted desired topologies %s", desired)
    return desired


def get_best_cpu_topology(flavor, image_meta, allow_threads=True,
                          numa_topology=None):
    """Identify best CPU topology for given constraints.

    Look at the properties set in the flavor extra specs and the image
    metadata and build up a list of all possible valid CPU topologies
    that can be used in the guest. Then return the best topology to use

    :param flavor: objects.Flavor instance to query extra specs from
    :param image_meta: nova.objects.ImageMeta object instance
    :param allow_threads: if the hypervisor supports CPU threads
    :param numa_topology: objects.InstanceNUMATopology instance that
                          may contain additional topology constraints
                          (such as threading information) that should
                          be considered

    :returns: an objects.VirtCPUTopology instance for best topology
    """
    return _get_desirable_cpu_topologies(flavor, image_meta,
                                         allow_threads, numa_topology)[0]


def _numa_cell_supports_pagesize_request(host_cell, inst_cell):
    """Determine whether the cell can accept the request.

    :param host_cell: host cell to fit the instance cell onto
    :param inst_cell: instance cell we want to fit

    :raises: exception.MemoryPageSizeNotSupported if custom page
             size not supported in host cell
    :returns: the page size able to be handled by host_cell
    """
    avail_pagesize = [page.size_kb for page in host_cell.mempages]
    avail_pagesize.sort(reverse=True)

    def verify_pagesizes(host_cell, inst_cell, avail_pagesize):
        inst_cell_mem = inst_cell.memory * units.Ki
        for pagesize in avail_pagesize:
            if host_cell.can_fit_hugepages(pagesize, inst_cell_mem):
                return pagesize

    if inst_cell.pagesize == MEMPAGES_SMALL:
        return verify_pagesizes(host_cell, inst_cell, avail_pagesize[-1:])
    elif inst_cell.pagesize == MEMPAGES_LARGE:
        return verify_pagesizes(host_cell, inst_cell, avail_pagesize[:-1])
    elif inst_cell.pagesize == MEMPAGES_ANY:
        return verify_pagesizes(host_cell, inst_cell, avail_pagesize)
    else:
        return verify_pagesizes(host_cell, inst_cell, [inst_cell.pagesize])


def _pack_instance_onto_cores(available_siblings,
                              instance_cell,
                              host_cell_id,
                              host_cell_shared_pcpu,
                              threads_per_core=1,
                              num_cpu_reserved=0, details=None):
    """Pack an instance onto a set of siblings.

    Calculate the pinning for the given instance and its topology,
    making sure that hyperthreads of the instance match up with those
    of the host when the pinning takes effect. Also ensure that the
    physical cores reserved for hypervisor on this host NUMA node do
    not break any thread policies.

    Currently the strategy for packing is to prefer siblings and try use
    cores evenly by using emptier cores first. This is achieved by the
    way we order cores in the sibling_sets structure, and the order in
    which we iterate through it.

    The main packing loop that iterates over the sibling_sets dictionary
    will not currently try to look for a fit that maximizes number of
    siblings, but will simply rely on the iteration ordering and picking
    the first viable placement.

    :param available_siblings: list of sets of CPU IDs corresponding to
                               available siblings per core
    :param instance_cell: An instance of objects.InstanceNUMACell
                          describing the pinning requirements of the
                          instance
    :param threads_per_core: number of threads per core in host's cell
    :param num_cpu_reserved: number of pCPUs reserved for hypervisor

    :returns: An instance of objects.InstanceNUMACell containing the
              pinning information, the physical cores reserved and
              potentially a new topology to be exposed to the
              instance. None if there is no valid way to satisfy the
              sibling requirements for the instance.

    """
    LOG.debug('Packing an instance onto a set of siblings: '
             '    available_siblings: %(siblings)s'
             '    instance_cell: %(cells)s'
             '    host_cell_id: %(host_cell_id)s'
             '    threads_per_core: %(threads_per_core)s',
                {'siblings': available_siblings,
                 'cells': instance_cell,
                 'host_cell_id': host_cell_id,
                 'threads_per_core': threads_per_core})

    # We build up a data structure that answers the question: 'Given the
    # number of threads I want to pack, give me a list of all the available
    # sibling sets (or groups thereof) that can accommodate it'
    sibling_sets = collections.defaultdict(list)
    for sib in available_siblings:
        for threads_no in range(1, len(sib) + 1):
            sibling_sets[threads_no].append(sib)
    LOG.debug('Built sibling_sets: %(siblings)s', {'siblings': sibling_sets})

    pinning = None
    threads_no = 1

    def _orphans(instance_cell, threads_per_core):
        """Number of instance CPUs which will not fill up a host core.

        Best explained by an example: consider set of free host cores as such:
            [(0, 1), (3, 5), (6, 7, 8)]
        This would be a case of 2 threads_per_core AKA an entry for 2 in the
        sibling_sets structure.

        If we attempt to pack a 5 core instance on it - due to the fact that we
        iterate the list in order, we will end up with a single core of the
        instance pinned to a thread "alone" (with id 6), and we would have one
        'orphan' vcpu.
        """
        return len(instance_cell) % threads_per_core

    def _threads(instance_cell, threads_per_core):
        """Threads to expose to the instance via the VirtCPUTopology.

        This is calculated by taking the GCD of the number of threads we are
        considering at the moment, and the number of orphans. An example for
            instance_cell = 6
            threads_per_core = 4

        So we can fit the instance as such:
            [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
              x  x  x  x    x  x

        We can't expose 4 threads, as that will not be a valid topology (all
        cores exposed to the guest have to have an equal number of threads),
        and 1 would be too restrictive, but we want all threads that guest sees
        to be on the same physical core, so we take GCD of 4 (max number of
        threads) and 2 (number of 'orphan' CPUs) and get 2 as the number of
        threads.
        """
        return fractions.gcd(threads_per_core, _orphans(instance_cell,
                                                        threads_per_core))

    def _get_pinning(threads_no, sibling_set, instance_cores,
                     num_cpu_reserved=0, details=None):
        """Determines pCPUs/vCPUs mapping

        Determines the pCPUs/vCPUs mapping regarding the number of
        threads which can be used per cores and pCPUs reserved.

        :param threads_no: Number of host threads per cores which can
                           be used to pin vCPUs according to the
                           policies :param sibling_set: List of
                           available threads per host cores on a
                           specific host NUMA node.
        :param instance_cores: Set of vCPUs requested.
        :param num_cpu_reserved: Number of additional host CPUs which
                                 needs to be reserved.

        NOTE: Depending on how host is configured (HT/non-HT) a thread can
              be considered as an entire core.
        """
        if threads_no * len(sibling_set) < (
                len(instance_cores) + num_cpu_reserved):
            # WRS: add details on failure to pin
            msg = ("NUMA %(N)d: CPUs requested %(R)d > avail %(A)d with "
                   "cpu thread policy %(P)s" %
                   {'N': host_cell_id,
                    'R': len(instance_cores),
                    'A': len(sibling_set) * threads_no,
                    'P': instance_cell.cpu_thread_policy})
            details = utils.details_append(details, msg)
            return None, None

        # Determines usable cores according the "threads number"
        # constraint.
        #
        # For a sibling_set=[(0, 1, 2, 3), (4, 5, 6, 7)] and thread_no 1:
        # usable_cores=[(0), (4),]
        #
        # For a sibling_set=[(0, 1, 2, 3), (4, 5, 6, 7)] and thread_no 2:
        # usable_cores=[(0, 1), (4, 5)]
        usable_cores = list(map(lambda s: list(s)[:threads_no], sibling_set))

        # Determines the mapping vCPUs/pCPUs based on the sets of
        # usable cores.
        #
        # For an instance_cores=[2, 3], usable_cores=[(0), (4)]
        # vcpus_pinning=[(2, 0), (3, 4)]
        vcpus_pinning = list(zip(sorted(instance_cores),
                                 itertools.chain(*usable_cores)))
        msg = ("Computed NUMA topology CPU pinning: usable pCPUs: "
               "%(usable_cores)s, vCPUs mapping: %(vcpus_pinning)s")
        msg_args = {
            'usable_cores': usable_cores,
            'vcpus_pinning': vcpus_pinning,
        }
        # WRS - demote to debug since this produces noise
        LOG.debug(msg, msg_args)

        cpuset_reserved = None
        if num_cpu_reserved:
            # Updates the pCPUs used based on vCPUs pinned to
            #
            # For vcpus_pinning=[(0, 2), (1, 3)], usable_cores=[(2, 3), (4, 5)]
            # usable_cores=[(), (4, 5)]
            for vcpu, pcpu in vcpus_pinning:
                for sib in usable_cores:
                    if pcpu in sib:
                        sib.remove(pcpu)

            # Determines the pCPUs reserved for hypervisor
            #
            # For usable_cores=[(), (4, 5)], num_cpu_reserved=1
            # cpuset_reserved=[4]
            cpuset_reserved = set(list(
                itertools.chain(*usable_cores))[:num_cpu_reserved])
            msg = ("Computed NUMA topology reserved pCPUs: usable pCPUs: "
                   "%(usable_cores)s, reserved pCPUs: %(cpuset_reserved)s")
            msg_args = {
                'usable_cores': usable_cores,
                'cpuset_reserved': cpuset_reserved,
            }
            # WRS - demote to debug since this produces noise
            LOG.debug(msg, msg_args)

        return vcpus_pinning, cpuset_reserved

    if (instance_cell.cpu_thread_policy ==
            fields.CPUThreadAllocationPolicy.REQUIRE):
        LOG.debug("Requested 'require' thread policy for %d cores",
                  len(instance_cell))
    elif (instance_cell.cpu_thread_policy ==
            fields.CPUThreadAllocationPolicy.PREFER):
        LOG.debug("Requested 'prefer' thread policy for %d cores",
                  len(instance_cell))
    elif (instance_cell.cpu_thread_policy ==
            fields.CPUThreadAllocationPolicy.ISOLATE):
        LOG.debug("Requested 'isolate' thread policy for %d cores",
                  len(instance_cell))
    else:
        LOG.debug("User did not specify a thread policy. Using default "
                  "for %d cores", len(instance_cell))

    if (instance_cell.cpu_thread_policy ==
            fields.CPUThreadAllocationPolicy.ISOLATE):
        # make sure we have at least one fully free core
        if threads_per_core not in sibling_sets:
            LOG.debug('Host does not have any fully free thread sibling sets.'
                      'It is not possible to emulate a non-SMT behavior '
                      'for the isolate policy without this.')
            # WRS: add details on failure to pin
            msg = ("NUMA %(N)d: Cannot use %(P)s cpu thread policy as there "
                   "are no CPUs with all siblings free" %
                   {'N': host_cell_id,
                    'P': instance_cell.cpu_thread_policy})
            details = utils.details_append(details, msg)
            return

        pinning, cpuset_reserved = _get_pinning(
            1,  # we only want to "use" one thread per core
            sibling_sets[threads_per_core],
            instance_cell.cpuset,
            num_cpu_reserved=num_cpu_reserved, details=details)
        # WRS: add details on failure to pin
        if not pinning:
            msg = ("NUMA %(N)d: Cannot use %(P)s cpu thread policy as "
                   "requested CPUs %(R)d > avail with all siblings "
                   "free %(A)d" %
                   {'N': host_cell_id,
                    'P': instance_cell.cpu_thread_policy,
                    'R': len(instance_cell),
                    'A': len(sibling_sets[threads_per_core])})
            details = utils.details_append(details, msg)
    else:  # REQUIRE, PREFER (explicit, implicit)
        # NOTE(ndipanov): We iterate over the sibling sets in descending order
        # of cores that can be packed. This is an attempt to evenly distribute
        # instances among physical cores
        for threads_no, sibling_set in sorted(
                (t for t in sibling_sets.items()), reverse=True):

            # NOTE(sfinucan): The key difference between the require and
            # prefer policies is that require will not settle for non-siblings
            # if this is all that is available. Enforce this by ensuring we're
            # using sibling sets that contain at least one sibling
            if (instance_cell.cpu_thread_policy ==
                    fields.CPUThreadAllocationPolicy.REQUIRE):
                if threads_no <= 1:
                    LOG.debug('Skipping threads_no: %s, as it does not satisfy'
                              ' the require policy', threads_no)
                    continue

            pinning, cpuset_reserved = _get_pinning(
                threads_no, sibling_set,
                instance_cell.cpuset,
                num_cpu_reserved=num_cpu_reserved, details=details)
            if pinning:
                break

        # NOTE(sfinucan): If siblings weren't available and we're using PREFER
        # (implicitly or explicitly), fall back to linear assignment across
        # cores
        if (instance_cell.cpu_thread_policy !=
                fields.CPUThreadAllocationPolicy.REQUIRE and
                not pinning):
            # WRS: add check to ensure there are enough available cpus
            if (len(instance_cell.cpuset) <=
                         len(list(itertools.chain(*sibling_set)))):
                pinning = list(zip(sorted(instance_cell.cpuset),
                              itertools.chain(*sibling_set)))

            # WRS: add details on failure to pin
            if not pinning:
                msg = ("NUMA %(N)d: CPUs requested %(R)d > avail %(A)d" %
                       {'N': host_cell_id,
                        'R': len(instance_cell.cpuset),
                        'A': len(list(itertools.chain(*sibling_set)))})
                details = utils.details_append(details, msg)

        threads_no = _threads(instance_cell, threads_no)

    if not pinning:
        return
    LOG.debug('Selected cores for pinning: %s, in cell %s', pinning,
                                                            host_cell_id)

    topology = objects.VirtCPUTopology(sockets=1,
                                       cores=len(pinning) // threads_no,
                                       threads=threads_no)
    instance_cell.pin_vcpus(*pinning)
    instance_cell.cpu_topology = topology
    instance_cell.id = host_cell_id
    instance_cell.cpuset_reserved = cpuset_reserved
    # WRS: add shared pcpu to cell
    instance_cell.shared_pcpu_for_vcpu = host_cell_shared_pcpu
    return instance_cell


def _numa_fit_instance_cell_with_pinning(host_cell, instance_cell,
                                         num_cpu_reserved=0,
                                         details=None):
    """Determine if cells can be pinned to a host cell.

    :param host_cell: objects.NUMACell instance - the host cell that
                      the instance should be pinned to
    :param instance_cell: objects.InstanceNUMACell instance without any
                          pinning information
    :param num_cpu_reserved: int - number of pCPUs reserved for hypervisor

    :returns: objects.InstanceNUMACell instance with pinning information,
              or None if instance cannot be pinned to the given host
    """

    details = utils.details_initialize(details=details)

    # WRS: exclude cells based on shared_pcpu mismatch
    if (instance_cell.shared_vcpu is not None and
        host_cell.shared_pcpu is None):
        msg = ("Shared vCPU not enabled on host cell %d, "
               "required by instance cell %d"
               % (host_cell.id, instance_cell.id))
        details = utils.details_append(details, msg)
        return

    # WRS: exclude shared_vcpu from cpuset if it exists
    instance_cell.cpuset.discard(instance_cell.shared_vcpu)

    # Check for an empty cpuset after removing the shared_vcpu
    if instance_cell.shared_vcpu is not None and not instance_cell.cpuset:
        msg = ("(numa:%(id)d shared vcpu with 0 requested "
               "dedicated vcpus is not allowed, avail:%(avail)d)" %
               {'id': host_cell.id,
                'avail': len(host_cell.cpuset)})
        details = utils.details_append(details, msg)
        return

    if host_cell.avail_cpus < len(instance_cell.cpuset):
        LOG.debug('Not enough available CPUs to schedule instance. '
                  'Oversubscription is not possible with pinned instances. '
                  'Required: %(required)s, actual: %(actual)s',
                  {'required': len(instance_cell.cpuset),
                   'actual': host_cell.avail_cpus})
        msg = "NUMA %d: CPUs avail(%d) < required(%d)" \
              % (host_cell.id, host_cell.avail_cpus,
                 len(instance_cell.cpuset))
        details = utils.details_append(details, msg)
        return

    required_cpus = len(instance_cell.cpuset) + num_cpu_reserved
    if host_cell.avail_cpus < required_cpus:
        LOG.debug('Not enough available CPUs to schedule instance. '
                  'Oversubscription is not possible with pinned instances. '
                  'Required: %(required)d (%(vcpus)d + %(num_cpu_reserved)d), '
                  'actual: %(actual)d',
                  {'required': required_cpus,
                   'vcpus': len(instance_cell.cpuset),
                   'actual': host_cell.avail_cpus,
                   'num_cpu_reserved': num_cpu_reserved})
        msg = "NUMA %d: CPUs avail(%d) < required(%d vcpus + %d reserved)" \
              % (host_cell.id, host_cell.avail_cpus,
                 len(instance_cell.cpuset), num_cpu_reserved)
        details = utils.details_append(details, msg)
        return

    if host_cell.avail_memory < instance_cell.memory:
        LOG.debug('Not enough available memory to schedule instance. '
                  'Oversubscription is not possible with pinned instances. '
                  'Required: %(required)s, actual: %(actual)s',
                  {'required': instance_cell.memory,
                   'actual': host_cell.memory})
        msg = "Memory avail(%d) < requested(%d)" \
              % (host_cell.avail_memory, instance_cell.memory)
        details = utils.details_append(details, msg)
        return

    if host_cell.siblings:
        LOG.debug('Using thread siblings for packing')
        # Try to pack the instance cell onto cores
        numa_cell = _pack_instance_onto_cores(
            host_cell.free_siblings, instance_cell, host_cell.id,
            host_cell.shared_pcpu,
            max(map(len, host_cell.siblings)),
            num_cpu_reserved=num_cpu_reserved, details=details)
    else:
        if (instance_cell.cpu_thread_policy ==
                fields.CPUThreadAllocationPolicy.REQUIRE):
            LOG.info("Host does not support hyperthreading or "
                     "hyperthreading is disabled, but 'require' "
                     "threads policy was requested.")
            msg = "Host does not support 'require' threads policy"
            details = utils.details_append(details, msg)
            return

        # Straightforward to pin to available cpus when there is no
        # hyperthreading on the host
        free_cpus = [set([cpu]) for cpu in host_cell.free_cpus]
        numa_cell = _pack_instance_onto_cores(
            free_cpus, instance_cell, host_cell.id, host_cell.shared_pcpu,
            num_cpu_reserved=num_cpu_reserved, details=details)

    if not numa_cell:
        LOG.debug('Failed to map instance cell CPUs to host cell CPUs')

    return numa_cell


def _numa_fit_instance_cell(host_cell, instance_cell, limit_cell=None,
                            cpuset_reserved=0, details=None):
    """Ensure an instance cell can fit onto a host cell

    Ensure an instance cell can fit onto a host cell and, if so, return
    a new objects.InstanceNUMACell with the id set to that of the host.
    Returns None if the instance cell exceeds the limits of the host.

    :param host_cell: host cell to fit the instance cell onto
    :param instance_cell: instance cell we want to fit
    :param limit_cell: an objects.NUMATopologyLimit or None
    :param cpuset_reserved: An int to indicate the number of CPUs overhead

    :returns: objects.InstanceNUMACell with the id set to that of the
              host, or None
    """
    LOG.debug('Attempting to fit instance cell %(cell)s on host_cell '
              '%(host_cell)s', {'cell': instance_cell, 'host_cell': host_cell})
    # NOTE (ndipanov): do not allow an instance to overcommit against
    # itself on any NUMA cell
    details = utils.details_initialize(details=details)
    if instance_cell.memory > host_cell.memory:
        LOG.debug('Not enough host cell memory to fit instance cell. '
                  'Required: %(required)d, actual: %(actual)d',
                  {'required': instance_cell.memory,
                   'actual': host_cell.memory})
        msg = 'Memory of instance(%d) > host_cell(%d) ' \
              % (instance_cell.memory, host_cell.memory)
        details = utils.details_append(details, msg)
        return None

    if len(instance_cell.cpuset) + cpuset_reserved > len(host_cell.cpuset):
        LOG.debug('Not enough host cell CPUs to fit instance cell. Required: '
                  '%(required)d + %(cpuset_reserved)d as overhead, '
                  'actual: %(actual)d',
                  {'required': len(instance_cell.cpuset),
                   'actual': len(host_cell.cpuset),
                   'cpuset_reserved': cpuset_reserved})
        msg = 'Cpu set of instance(%d) > host_cell(%d) ' \
              % (len(instance_cell.cpuset),
                 len(host_cell.cpuset))
        details = utils.details_append(details, msg)
        return None

    # WRS: if numa pinning requested confirm correct host numa cell
    if instance_cell.numa_pinning_requested:
        if instance_cell.physnode != host_cell.id:
            msg = "Host NUMA: %d excluded, does not match requested NUMA: %d" \
                  % (host_cell.id, instance_cell.physnode)
            details = utils.details_append(details, msg)
            return None

    if instance_cell.cpu_pinning_requested:
        LOG.debug('Pinning has been requested')
        new_instance_cell = _numa_fit_instance_cell_with_pinning(
            host_cell, instance_cell, cpuset_reserved, details=details)
        if not new_instance_cell:
            return None
        new_instance_cell.pagesize = instance_cell.pagesize
        instance_cell = new_instance_cell

    elif limit_cell:
        LOG.debug('No pinning requested, considering limitations on usable cpu'
                  ' and memory')
        memory_usage = host_cell.memory_usage + instance_cell.memory
        cpu_usage = (host_cell.cpu_usage + len(instance_cell.cpuset) /
                     limit_cell.cpu_allocation_ratio)
        cpu_limit = len(host_cell.cpuset)
        ram_limit = host_cell.memory * limit_cell.ram_allocation_ratio
        if memory_usage > ram_limit or cpu_usage > cpu_limit:
            if memory_usage > ram_limit:
                LOG.debug('Host cell has limitations on usable memory. There '
                          'is not enough free memory to schedule this '
                          'instance. Usage: %(usage)d, limit: %(limit)d',
                          {'usage': memory_usage, 'limit': ram_limit})
                msg = 'limits: Not enough memory'
                details = utils.details_append(details, msg)
            if cpu_usage > cpu_limit:
                LOG.debug('Host cell has limitations on usable CPUs. There '
                          'are not enough free CPUs to schedule this '
                          'instance. Usage: %(usage)d, limit: %(limit)d',
                          {'usage': cpu_usage, 'limit': cpu_limit})
                msg = 'limits: Not enough cpus'
                details = utils.details_append(details, msg)
            return None

    pagesize = None
    if instance_cell.pagesize:
        pagesize = _numa_cell_supports_pagesize_request(
            host_cell, instance_cell)
        if not pagesize:
            LOG.debug('Host does not support requested memory pagesize. '
                      'Requested: %d kB', instance_cell.pagesize)

            def page_to_readable_units(pgsize):
                unit = 'K'
                size = pgsize
                if pgsize >= units.Ki and pgsize < units.Mi:
                    unit = 'M'
                    size = pgsize / units.Ki
                elif pgsize >= units.Mi:
                    unit = 'G'
                    size = pgsize / units.Mi
                return {'unit': unit, 'size': size}

            def avail_pagesizes_mem(cell):
                mem = []
                for mempage in cell.mempages:
                    ret = page_to_readable_units(mempage.size_kb)
                    m = '%(sz)s%(U)s: %(A).0f MiB' % \
                        {'sz': ret['size'], 'U': ret['unit'],
                         'A': mempage.size_kb * mempage.free / units.Ki,
                        }
                    mem.append(m)
                return mem

            if instance_cell.pagesize == MEMPAGES_SMALL:
                pgrequest = 'small'
                msg = "Not enough memory or not pagesize divisible"
            elif instance_cell.pagesize == MEMPAGES_LARGE:
                pgrequest = 'large'
                msg = "Not enough memory or not pagesize divisible"
            elif instance_cell.pagesize == MEMPAGES_ANY:
                pgrequest = 'any'
                msg = "Not enough memory or not pagesize divisible"
            else:
                ret = page_to_readable_units(instance_cell.pagesize)
                pgrequest = "%(sz)s%(U)s" % {'sz': ret['size'],
                            'U': ret['unit']}
                if divmod(instance_cell.memory * units.Ki,
                          instance_cell.pagesize)[1] > 0:
                    det = "Not pagesize divisible: numa:%(n)d " \
                          "page: %(pg)s size: %(A).0f MiB" % \
                          {'n': host_cell.id, 'pg': pgrequest,
                           'A': instance_cell.memory}
                    details = utils.details_append(details, det)
                    return None
                msg = "Not enough memory"

            m = avail_pagesizes_mem(host_cell)
            det = "%(msg)s: numa:%(node)d req: %(pg)s:%(A).0f MiB, " \
                  "(avail: %(m)s)" % \
                  {'msg': msg, 'node': host_cell.id,
                   'pg': pgrequest, 'A': instance_cell.memory,
                   'm': '; '.join(m)}
            details = utils.details_append(details, det)
            return None

    # L3 CAT Support
    if ((not host_cell.has_cachetune) and instance_cell.cachetune_requested):
        msg = "cache allocation technology not supported"
        details = utils.details_append(details, msg)
        return None

    if (instance_cell.cachetune_requested and
            (not instance_cell.cpu_pinning_requested)):
        msg = ("L3 cache request requires hw:cpu_policy=%(policy)s" %
               {'policy': fields.CPUAllocationPolicy.DEDICATED})
        details = utils.details_append(details, msg)
        return None

    if (host_cell.has_cachetune and instance_cell.cachetune_requested):
        # CDP vs unified host check
        m = []
        if host_cell.has_cachetune_cdp:
            cachetune_type = 'cdp'
            if instance_cell.l3_both_size is not None:
                m.append('both:%r' % (instance_cell.l3_both_size))
        else:
            cachetune_type = 'unified'
            if (instance_cell.l3_code_size is not None):
                m.append('code:%s' % (instance_cell.l3_code_size))
            if (instance_cell.l3_data_size is not None):
                m.append('data:%s' % (instance_cell.l3_data_size))
        if m:
            msg = ("L3 cache request (%(R)s) not supported on '%(T)s' host" %
                   {'R': ', '.join(m),
                    'T': cachetune_type})
            details = utils.details_append(details, msg)
            return None

        # Cache size check
        cache_size = sum(utils.roundup(x, host_cell.l3_granularity)
                         for x in (instance_cell.l3_both_size,
                                   instance_cell.l3_code_size,
                                   instance_cell.l3_data_size)
                         if x is not None)
        max_alloc = host_cell.max_l3_allocation
        if cache_size > max_alloc:
            msg = ("L3 cache request (%(R)d) > max supported "
                   "allocation (%(M)d)" %
                   {'R': cache_size,
                    'M': max_alloc})
            details = utils.details_append(details, msg)
            return None

        cache_avail = host_cell.avail_cache
        if cache_size > cache_avail:
            msg = ("NUMA %(N)d: L3 cache requested %(R)d > avail %(A)d KiB" %
                   {'N': host_cell.id,
                    'R': cache_size,
                    'A': cache_avail})
            details = utils.details_append(details, msg)
            return None

    instance_cell.id = host_cell.id
    instance_cell.pagesize = pagesize
    return instance_cell


def _get_flavor_image_meta(key, flavor, image_meta):
    """Extract both flavor- and image-based variants of metadata."""
    flavor_key = ':'.join(['hw', key])
    image_key = '_'.join(['hw', key])

    flavor_policy = flavor.get('extra_specs', {}).get(flavor_key)
    image_policy = image_meta.properties.get(image_key)

    return flavor_policy, image_policy


def _numa_get_pagesize_constraints(flavor, image_meta):
    """Return the requested memory page size

    :param flavor: a Flavor object to read extra specs from
    :param image_meta: nova.objects.ImageMeta object instance

    :raises: MemoryPageSizeInvalid if flavor extra spec or image
             metadata provides an invalid hugepage value
    :raises: MemoryPageSizeForbidden if flavor extra spec request
             conflicts with image metadata request
    :returns: a page size requested or MEMPAGES_*
    """

    def check_and_return_pages_size(request):
        if request == "any":
            return MEMPAGES_ANY
        elif request == "large":
            return MEMPAGES_LARGE
        elif request == "small":
            return MEMPAGES_SMALL
        else:
            try:
                request = int(request)
            except ValueError:
                try:
                    request = strutils.string_to_bytes(
                        request, return_int=True) / units.Ki
                except ValueError:
                    request = 0

        if request <= 0:
            raise exception.MemoryPageSizeInvalid(pagesize=request)

        return request

    flavor_request, image_request = _get_flavor_image_meta(
        'mem_page_size', flavor, image_meta)

    if not flavor_request and image_request:
        raise exception.MemoryPageSizeForbidden(
            pagesize=image_request,
            against="<empty>")

    if not flavor_request:
        # Nothing was specified for hugepages,
        # let's the default process running.

        # WRS - Set default memory pagesize.
        # This has side-effect of forcing VMs to a numa node.
        if CONF.default_mempages_size:
            flavor_request = CONF.default_mempages_size
            LOG.info("defaulting pagesize to: %s", flavor_request)
        else:
            return None

    pagesize = check_and_return_pages_size(flavor_request)
    if image_request:
        img_pagesize = check_and_return_pages_size(image_request)
        if (pagesize in (MEMPAGES_ANY, MEMPAGES_LARGE)
                or pagesize == img_pagesize):

            return img_pagesize
        else:
            raise exception.MemoryPageSizeForbidden(
                pagesize=image_request,
                against=flavor_request)

    return pagesize


def _numa_get_flavor_cpu_map_list(flavor):
    hw_numa_cpus = []
    extra_specs = flavor.get("extra_specs", {})
    for cellid in range(objects.ImageMetaProps.NUMA_NODES_MAX):
        cpuprop = "hw:numa_cpus.%d" % cellid
        if cpuprop not in extra_specs:
            break
        hw_numa_cpus.append(
            parse_cpu_spec(extra_specs[cpuprop]))

    if hw_numa_cpus:
        return hw_numa_cpus


def _numa_get_cpu_map_list(flavor, image_meta):
    flavor_cpu_list = _numa_get_flavor_cpu_map_list(flavor)
    image_cpu_list = image_meta.properties.get("hw_numa_cpus", None)

    if flavor_cpu_list is None:
        return image_cpu_list
    else:
        if image_cpu_list is not None:
            raise exception.ImageNUMATopologyForbidden(
                name='hw_numa_cpus')
        return flavor_cpu_list


def _numa_get_flavor_mem_map_list(flavor):
    hw_numa_mem = []
    extra_specs = flavor.get("extra_specs", {})
    for cellid in range(objects.ImageMetaProps.NUMA_NODES_MAX):
        memprop = "hw:numa_mem.%d" % cellid
        if memprop not in extra_specs:
            break
        hw_numa_mem.append(int(extra_specs[memprop]))

    if hw_numa_mem:
        return hw_numa_mem


def _numa_get_mem_map_list(flavor, image_meta):
    flavor_mem_list = _numa_get_flavor_mem_map_list(flavor)
    image_mem_list = image_meta.properties.get("hw_numa_mem", None)

    if flavor_mem_list is None:
        return image_mem_list
    else:
        if image_mem_list is not None:
            raise exception.ImageNUMATopologyForbidden(
                name='hw_numa_mem')
        return flavor_mem_list


def _get_cpu_policy_constraints(flavor, image_meta, numa_topology=None):
    """Validate and return the requested CPU policy."""
    flavor_policy, image_policy = _get_flavor_image_meta(
        'cpu_policy', flavor, image_meta)

    if flavor_policy == fields.CPUAllocationPolicy.DEDICATED:
        cpu_policy = flavor_policy
    elif flavor_policy == fields.CPUAllocationPolicy.SHARED:
        if image_policy == fields.CPUAllocationPolicy.DEDICATED:
            raise exception.ImageCPUPinningForbidden()
        cpu_policy = flavor_policy
    elif image_policy == fields.CPUAllocationPolicy.DEDICATED:
        cpu_policy = image_policy
    else:
        cpu_policy = fields.CPUAllocationPolicy.SHARED

    # WRS: If NUMA pinning is enabled, then implicitly enable CPU pinning
    # unless it has been explicitly disabled.
    if numa_topology and numa_topology.cells[0].physnode is not None:
        if (flavor_policy == fields.CPUAllocationPolicy.SHARED or
                image_policy == fields.CPUAllocationPolicy.SHARED):
            raise exception.ImageNUMATopologyNodesForbidden()
        else:
            cpu_policy = fields.CPUAllocationPolicy.DEDICATED

    return cpu_policy


def _get_cpu_thread_policy_constraints(flavor, image_meta):
    """Validate and return the requested CPU thread policy."""
    flavor_policy, image_policy = _get_flavor_image_meta(
        'cpu_thread_policy', flavor, image_meta)

    if flavor_policy in [None, fields.CPUThreadAllocationPolicy.PREFER]:
        policy = flavor_policy or image_policy
    elif image_policy and image_policy != flavor_policy:
        raise exception.ImageCPUThreadPolicyForbidden()
    else:
        policy = flavor_policy

    return policy


def _get_extra_specs_l3_cache_vcpu_map_list(extra_specs):
    """Return a list of L3 cache vcpus (vcpu_map) per numa node

    :param extra_specs dictionary

    :raises: None
    :returns: list of L3 cache vcpus (vcpu_map) per numa node
    """
    vcpus = []
    for cellid in range(objects.ImageMetaProps.NUMA_NODES_MAX):
        key = "hw:cache_vcpus.%d" % cellid
        if key not in extra_specs:
            break
        vcpus.append(parse_cpu_spec(extra_specs[key]))

    if vcpus:
        return vcpus


def get_l3_cache_vcpu_map_list(extra_specs, image_props):
    """Return a list of L3 cache vcpus (vcpu_map) per numa node

    :param extra_specs dictionary
    :param image_props dictionary

    :raises: ImageNUMATopologyForbidden
    :returns: list of L3 cache vcpus (vcpu_map) per numa node
    """
    extra_vcpus = _get_extra_specs_l3_cache_vcpu_map_list(extra_specs)
    image_vcpus = image_props.get("hw_cache_vcpus", None)
    # TODO(jgauld): Make hardware properties check as strict as extra-specs.

    if extra_vcpus is None:
        return image_vcpus
    else:
        if image_vcpus is not None:
            raise exception.ImageNUMATopologyForbidden(
                name='hw_cache_vcpus')
        return extra_vcpus


def _get_extra_specs_l3_cache_size_KiB_list(extra_specs, prefix):
    """Return a list of L3 cache size (KiB) per numa node
    :param extra_specs dictionary
    :param key prefix string

    :raises: none
    :returns: list of L3 cache size (KiB) per numa node
    """
    size_KiB = []

    for cellid in range(objects.ImageMetaProps.NUMA_NODES_MAX):
        key = "%s.%d" % (prefix, cellid)
        if key not in extra_specs:
            break
        size_KiB.append(int(extra_specs[key]))

    if size_KiB:
        return size_KiB


def get_l3_cache_size_KiB_list(extra_specs, image_props, prefix):
    """Return a list of L3 cache size (KiB) per numa node

    :param extra_specs dictionary
    :param image_props dictionary
    :param key prefix string, eg, 'hw:cache_l3'

    :raises: ImageNUMATopologyForbidden
    :returns: list of L3 cache size (KiB) per numa node
    """
    extra_size_KiB = _get_extra_specs_l3_cache_size_KiB_list(extra_specs,
                                                             prefix)
    hw_prefix = prefix.replace(":", "_")
    image_size_KiB = image_props.get(hw_prefix, None)
    # TODO(jgauld): Make hardware properties check as strict as extra-specs.

    if extra_size_KiB is None:
        return image_size_KiB
    else:
        if image_size_KiB is not None:
            raise exception.ImageNUMATopologyForbidden(
                name=hw_prefix)
        return extra_size_KiB


# WRS: get numa node list from hw:numa_node.X extra spec
def _numa_get_flavor_node_map_list(flavor):
    hw_numa_node = []
    hw_numa_node_set = False
    extra_specs = flavor.get("extra_specs", {})
    for cellid in range(objects.ImageMetaProps.NUMA_NODES_MAX):
        nodeprop = "hw:numa_node.%d" % cellid
        if nodeprop not in extra_specs:
            break
        hw_numa_node.append(int(extra_specs[nodeprop]))
        hw_numa_node_set = True

    if hw_numa_node_set:
        return hw_numa_node


# WRS: get numa node list from flavor or image
def _numa_get_node_map_list(flavor, image_meta):
    flavor_node_list = _numa_get_flavor_node_map_list(flavor)
    image_node_list = image_meta.properties.get("hw_numa_node", None)

    if flavor_node_list is None:
        return image_node_list
    else:
        if image_node_list is not None:
            raise exception.ImageNUMATopologyForbidden(
                name='hw_numa_node')
        return flavor_node_list


# WRS: add node_list
def _numa_get_constraints_manual(nodes, flavor, cpu_list, mem_list,
                                 node_list):
    cells = []
    totalmem = 0

    availcpus = set(range(flavor.vcpus))

    for node in range(nodes):
        mem = mem_list[node]
        cpuset = cpu_list[node]
        physnode = node_list[node] if node_list else None

        for cpu in cpuset:
            if cpu > (flavor.vcpus - 1):
                raise exception.ImageNUMATopologyCPUOutOfRange(
                    cpunum=cpu, cpumax=(flavor.vcpus - 1))

            if cpu not in availcpus:
                raise exception.ImageNUMATopologyCPUDuplicates(
                    cpunum=cpu)

            availcpus.remove(cpu)

        cells.append(objects.InstanceNUMACell(
            id=node, cpuset=cpuset, memory=mem, physnode=physnode))
        totalmem = totalmem + mem

    if availcpus:
        raise exception.ImageNUMATopologyCPUsUnassigned(
            cpuset=str(availcpus))

    if totalmem != flavor.memory_mb:
        raise exception.ImageNUMATopologyMemoryOutOfRange(
            memsize=totalmem,
            memtotal=flavor.memory_mb)

    return objects.InstanceNUMATopology(cells=cells)


def is_realtime_enabled(flavor):
    flavor_rt = flavor.get('extra_specs', {}).get("hw:cpu_realtime")
    return strutils.bool_from_string(flavor_rt)


def _get_realtime_mask(flavor, image):
    """Returns realtime mask based on flavor/image meta"""
    flavor_mask, image_mask = _get_flavor_image_meta(
        'cpu_realtime_mask', flavor, image)

    # Image masks are used ahead of flavor masks as they will have more
    # specific requirements
    return image_mask or flavor_mask


def vcpus_realtime_topology(flavor, image):
    """Determines instance vCPUs used as RT for a given spec"""
    mask = _get_realtime_mask(flavor, image)
    if not mask:
        raise exception.RealtimeMaskNotFoundOrInvalid()

    vcpus_rt = parse_cpu_spec("0-%d,%s" % (flavor.vcpus - 1, mask))
    if len(vcpus_rt) < 1:
        raise exception.RealtimeMaskNotFoundOrInvalid()

    # WRS: extended realtime validation
    vcpus_set = set(range(flavor.vcpus))
    vcpus_em = vcpus_set - vcpus_rt

    # Determine the specific key used to define the mask, noting that mask is
    # evaluated (image_mask or flavor_mask).
    ikey = 'hw_cpu_realtime_mask'
    fkey = 'hw:cpu_realtime_mask'
    key = None
    if ikey in image.properties:
        key = ikey
    if fkey in flavor.extra_specs:
        key = fkey

    # Check that vcpu_rt, vcpus_em are within valid range
    if ((not vcpus_rt.issubset(vcpus_set)) or
            (not vcpus_em.issubset(vcpus_set))):
        msg = (_('%(K)s (%(V)s) must be a subset of vCPUs (%(S)s)') %
               {'K': key,
                'V': mask,
                'S': utils.list_to_range(list(vcpus_set))
                })
        raise exception.RealtimeMaskNotFoundOrInvalid(msg)

    # Check that we have realtime vCPUs
    if not vcpus_rt:
        msg = (_('%(K)s (%(V)s) does not have realtime vCPUS defined') %
               {'K': key, 'V': mask})
        raise exception.RealtimeMaskNotFoundOrInvalid(msg)

    # Check that we have normal vCPUs
    if not vcpus_em:
        msg = (_('%(K)s (%(V)s) does not have normal vCPUS defined') %
               {'K': key, 'V': mask})
        raise exception.RealtimeMaskNotFoundOrInvalid(msg)

    # Check that hw:wrs:shared_vcpu is a subset of non-realtime vcpus
    shared_vcpu = flavor.get('extra_specs', {}).get("hw:wrs:shared_vcpu", None)
    if shared_vcpu is not None:
        shared_vcpus = set([int(shared_vcpu)])
        if not shared_vcpus.issubset(vcpus_em):
            msg = (_("hw:wrs:shared_vcpu (%(S)s) is not a subset of "
                     "non-realtime vCPUs (%(N)s)") %
                   {"S": utils.list_to_range(list(shared_vcpus)),
                    "N": utils.list_to_range(list(vcpus_em))})
            raise exception.RealtimeMaskNotFoundOrInvalid(msg)

    return vcpus_rt


# WRS: extension
def _get_pci_affinity_mask(flavor):
    """Parse pci affinity mask based on flavor extra-spec.

    Returns set of vcpu ids with corresponding pci irq affinity mask.
    """
    flavor_map = flavor.get('extra_specs', {}).get("hw:pci_irq_affinity_mask")
    if not flavor_map:
        return None

    cpuset_ids = parse_cpu_spec(flavor_map)
    if not cpuset_ids:
        raise exception.Invalid(_("No CPUs available after parsing %r") %
                                flavor_map)
    return cpuset_ids


# WRS: add node_list
def _numa_get_constraints_auto(nodes, flavor, node_list):
    if ((flavor.vcpus % nodes) > 0 or
        (flavor.memory_mb % nodes) > 0):
        raise exception.ImageNUMATopologyAsymmetric()

    cells = []
    for node in range(nodes):
        ncpus = int(flavor.vcpus / nodes)
        mem = int(flavor.memory_mb / nodes)
        start = node * ncpus
        cpuset = set(range(start, start + ncpus))
        physnode = node_list[node] if node_list else None

        cells.append(objects.InstanceNUMACell(
            id=node, cpuset=cpuset, memory=mem, physnode=physnode))

    return objects.InstanceNUMATopology(cells=cells)


def get_emulator_threads_constraint(flavor, image_meta):
    """Determines the emulator threads policy"""
    emu_threads_policy = flavor.get('extra_specs', {}).get(
        'hw:emulator_threads_policy')
    LOG.debug("emulator threads policy constraint: %s", emu_threads_policy)

    if not emu_threads_policy:
        return

    if emu_threads_policy not in fields.CPUEmulatorThreadsPolicy.ALL:
        raise exception.InvalidEmulatorThreadsPolicy(
            requested=emu_threads_policy,
            available=str(fields.CPUEmulatorThreadsPolicy.ALL))

    if emu_threads_policy == fields.CPUEmulatorThreadsPolicy.ISOLATE:
        # In order to make available emulator threads policy, a dedicated
        # CPU policy is necessary.
        cpu_policy = _get_cpu_policy_constraints(flavor, image_meta)
        if cpu_policy != fields.CPUAllocationPolicy.DEDICATED:
            raise exception.BadRequirementEmulatorThreadsPolicy()

    return emu_threads_policy


def _validate_numa_nodes(nodes):
    """Validate NUMA nodes number

    :param nodes: number of NUMA nodes

    :raises: exception.InvalidNUMANodesNumber if the number of NUMA
             nodes is less than 1 or not an integer
    """
    if nodes is not None and (not strutils.is_int_like(nodes) or
       int(nodes) < 1):
        raise exception.InvalidNUMANodesNumber(nodes=nodes)


# L3 CAT Support
def numa_l3_cache_get_constraints(flavor, image_meta, nodes):
    """Create lists of L3 cache related info per numa node,
       and raise exceptions for insuffient or invalid specification.

    :param flavor: Flavor object to read extra specs from
    :param image_meta: nova.objects.ImageMeta object instance
    :param nodes: number of numa nodes

    Raises exception.ImageL3CacheIncomplete() if insufficient
    parameters are specified for CAT.

    Raises exception.ImageL3CacheInvalid() if incorrect
    parameters are specified for CAT, eg. both unified and CDP.

    :return: cache_vcpus : list of L3 cache vcpus per node (or None),
             both_size   : list of L3 unified cache size per node (or None),
             code_size   : list of L3 CDP cache code size per node (or None),
             data_size   : list of L3 CDP cache data size per node (or None)
    """
    extra_specs = flavor.extra_specs
    image_props = image_meta.properties

    cache_vcpus = get_l3_cache_vcpu_map_list(
        extra_specs, image_props)
    if cache_vcpus is not None:
        wants_vcpus = True
    else:
        wants_vcpus = False

    both_size = get_l3_cache_size_KiB_list(
        extra_specs, image_props, 'hw:cache_l3')
    if both_size is not None:
        wants_both = True
    else:
        wants_both = False

    code_size = get_l3_cache_size_KiB_list(
        extra_specs, image_props, 'hw:cache_l3_code')
    if code_size is not None:
        wants_code = True
    else:
        wants_code = False

    data_size = get_l3_cache_size_KiB_list(
        extra_specs, image_props, 'hw:cache_l3_data')
    if data_size is not None:
        wants_data = True
    else:
        wants_data = False

    if wants_code or wants_data:
        wants_cdp = True
    else:
        wants_cdp = False

    # Cannot specify hw:cache_l3.x with either hw:cache_l3_code.x or
    # hw:cache_l3_data.x
    if [wants_both, wants_cdp].count(True) == 2:
        raise exception.ImageL3CacheInvalid(
            name='hw:cache_l3.x, hw:cache_l3_code.x, hw:cache_l3_data.x')

    # Cannot specify just hw:cache_l3_code.x or just hw:cache_l3_data.x
    if [wants_code, wants_data].count(True) == 1:
        raise exception.ImageL3CacheIncomplete(
            name='hw:cache_l3_code.x, hw:cache_l3_data.x')

    # Cannot specify hw:cache_vcpus.x without hw:cache_l3.x, or
    # hw:cache_l3_code.x, hw:cache_l3_data.x
    if (wants_vcpus and
            ([wants_both, wants_code, wants_data].count(True) == 0)):
        raise exception.ImageL3CacheIncomplete(
            name='hw:cache_vcpus.x and hw:cache_l3.x, or '
                 'hw:cache_vcpus.x, hw:cache_l3_code.x, and '
                 'hw:cache_l3_data.x')

    # If any node has data set, all nodes must have data set
    if cache_vcpus is not None and len(cache_vcpus) != nodes:
        raise exception.ImageL3CacheIncomplete(name='hw:cache_vcpus.x')
    if both_size is not None and len(both_size) != nodes:
        raise exception.ImageL3CacheIncomplete(name='hw:cache_l3.x')
    if code_size is not None and len(code_size) != nodes:
        raise exception.ImageL3CacheIncomplete(name='hw:cache_l3_code.x')
    if data_size is not None and len(data_size) != nodes:
        raise exception.ImageL3CacheIncomplete(name='hw:cache_l3_data.x')

    return (cache_vcpus, both_size, code_size, data_size)


# TODO(sahid): Move numa related to hardward/numa.py
def numa_get_constraints(flavor, image_meta):
    """Return topology related to input request.

    :param flavor: a flavor object to read extra specs from
    :param image_meta: nova.objects.ImageMeta object instance

    :raises: exception.InvalidNUMANodesNumber if the number of NUMA
             nodes is less than 1 or not an integer
    :raises: exception.ImageNUMATopologyForbidden if an attempt is made
             to override flavor settings with image properties
    :raises: exception.MemoryPageSizeInvalid if flavor extra spec or
             image metadata provides an invalid hugepage value
    :raises: exception.MemoryPageSizeForbidden if flavor extra spec
             request conflicts with image metadata request
    :raises: exception.ImageNUMATopologyIncomplete if the image
             properties are not correctly specified
    :raises: exception.ImageNUMATopologyAsymmetric if the number of
             NUMA nodes is not a factor of the requested total CPUs or
             memory
    :raises: exception.ImageNUMATopologyCPUOutOfRange if an instance
             CPU given in a NUMA mapping is not valid
    :raises: exception.ImageNUMATopologyCPUDuplicates if an instance
             CPU is specified in CPU mappings for two NUMA nodes
    :raises: exception.ImageNUMATopologyCPUsUnassigned if an instance
             CPU given in a NUMA mapping is not assigned to any NUMA node
    :raises: exception.ImageNUMATopologyMemoryOutOfRange if sum of memory from
             each NUMA node is not equal with total requested memory
    :raises: exception.ImageCPUPinningForbidden if a CPU policy
             specified in a flavor conflicts with one defined in image
             metadata
    :raises: exception.RealtimeConfigurationInvalid if realtime is
             requested but dedicated CPU policy is not also requested
    :raises: exception.RealtimeMaskNotFoundOrInvalid if realtime is
             requested but no mask provided
    :raises: exception.CPUThreadPolicyConfigurationInvalid if a CPU thread
             policy conflicts with CPU allocation policy
    :raises: exception.ImageCPUThreadPolicyForbidden if a CPU thread policy
             specified in a flavor conflicts with one defined in image metadata
    :returns: objects.InstanceNUMATopology, or None
    """
    flavor_nodes, image_nodes = _get_flavor_image_meta(
        'numa_nodes', flavor, image_meta)
    if flavor_nodes and image_nodes:
        raise exception.ImageNUMATopologyForbidden(
            name='hw_numa_nodes')

    nodes = None
    if flavor_nodes:
        _validate_numa_nodes(flavor_nodes)
        nodes = int(flavor_nodes)
    else:
        _validate_numa_nodes(image_nodes)
        nodes = image_nodes

    pagesize = _numa_get_pagesize_constraints(
        flavor, image_meta)

    # WRS: Need to check if we want numa pinning since we allow
    # pinning a single implicit numa node.
    node_list = _numa_get_node_map_list(flavor, image_meta)

    numa_topology = None
    if nodes or pagesize or node_list:
        nodes = nodes or 1

        cpu_list = _numa_get_cpu_map_list(flavor, image_meta)
        mem_list = _numa_get_mem_map_list(flavor, image_meta)

        # L3 CAT Support
        cache_vcpus, both_size, code_size, data_size = \
            numa_l3_cache_get_constraints(flavor, image_meta, nodes)

        # If one property list is specified for cpu/mem then both must be.
        # The physnode is optional.
        if ((cpu_list is None and mem_list is not None) or
            (cpu_list is not None and mem_list is None)):
            raise exception.ImageNUMATopologyIncomplete()

        # If any node has data set, all nodes must have data set
        if ((cpu_list is not None and len(cpu_list) != nodes) or
            (mem_list is not None and len(mem_list) != nodes)):
            raise exception.ImageNUMATopologyIncomplete()

        # WRS: Special test for numa nodes, because they can be specified
        # independently of setting CPUs/RAM.
        if node_list is not None and len(node_list) != nodes:
            raise exception.ImageNUMATopologyNodesIncomplete()

        # WRS: A bit of paranoia here.
        # host nodes must be all specified or all unspecified.
        if node_list and node_list.count(None) != len(node_list):
            # If any are not None then they all must be.
            if any(node is None for node in node_list):
                raise exception.ImageNUMATopologyNodesIncomplete()
            # Instance nodes must be pinned to separate host nodes.
            if len(node_list) != len(set(node_list)):
                raise exception.ImageNUMATopologyNodesDuplicates()

        if cpu_list is None:
            numa_topology = _numa_get_constraints_auto(
                nodes, flavor, node_list)
        else:
            numa_topology = _numa_get_constraints_manual(
                nodes, flavor, cpu_list, mem_list, node_list)

        # L3 CAT support
        # Populate instance numa_topology L3 CAT fields.
        for node, cell in enumerate(numa_topology.cells):
            if any(x is not None for x in (both_size, code_size, data_size)):
                if cache_vcpus is not None:
                    cell.l3_cpuset = set(cache_vcpus[node])
                else:
                    cell.l3_cpuset = set(cell.cpuset)
            if both_size is not None:
                cell.l3_both_size = both_size[node]
            if code_size is not None:
                cell.l3_code_size = code_size[node]
            if data_size is not None:
                cell.l3_data_size = data_size[node]

        # We currently support same pagesize for all cells.
        for c in numa_topology.cells:
            setattr(c, 'pagesize', pagesize)

    cpu_policy = _get_cpu_policy_constraints(flavor, image_meta,
                                             numa_topology=numa_topology)
    cpu_thread_policy = _get_cpu_thread_policy_constraints(flavor, image_meta)
    rt_mask = _get_realtime_mask(flavor, image_meta)
    emu_thread_policy = get_emulator_threads_constraint(flavor, image_meta)

    # sanity checks

    rt = is_realtime_enabled(flavor)

    if rt and cpu_policy != fields.CPUAllocationPolicy.DEDICATED:
        raise exception.RealtimeConfigurationInvalid()

    if rt and not rt_mask:
        raise exception.RealtimeMaskNotFoundOrInvalid()

    if cpu_policy == fields.CPUAllocationPolicy.SHARED:
        if cpu_thread_policy:
            raise exception.CPUThreadPolicyConfigurationInvalid()
        return numa_topology

    # WRS: If cpu pinning is requested we can check for hw:wrs:shared_vcpu
    # This means we "un-pin" the shared_vcpu index
    shared_vcpu = flavor.get('extra_specs', {}).get("hw:wrs:shared_vcpu", None)
    if shared_vcpu is not None:
        shared_vcpu = int(shared_vcpu)

    if numa_topology:
        for cell in numa_topology.cells:
            cell.cpu_policy = cpu_policy
            cell.cpu_thread_policy = cpu_thread_policy
            # WRS: For multiple cells, we add shared_vcpu to only one
            if shared_vcpu in cell.cpuset:
                cell.cpuset.discard(shared_vcpu)
                cell.shared_vcpu = shared_vcpu
            else:
                cell.shared_vcpu = None
    else:
        cpu_set = set(range(flavor.vcpus))
        cpu_set.discard(shared_vcpu)
        single_cell = objects.InstanceNUMACell(
            id=0,
            cpuset=cpu_set,
            shared_vcpu=shared_vcpu,
            memory=flavor.memory_mb,
            cpu_policy=cpu_policy,
            cpu_thread_policy=cpu_thread_policy)
        numa_topology = objects.InstanceNUMATopology(cells=[single_cell])

    if emu_thread_policy:
        numa_topology.emulator_threads_policy = emu_thread_policy

    return numa_topology


# WRS: This code/loop was refactored from numa_fit_instance_to_host().
def _numa_fit_instance_to_host(host_topology, instance_topology, limits,
                               details, pci_requests, pci_stats,
                               pci_strict=True):

    emulator_threads_policy = None
    if 'emulator_threads_policy' in instance_topology:
        emulator_threads_policy = instance_topology.emulator_threads_policy

    # If PCI device(s) are not required, prefer host cells that don't have
    # devices attached. Presence of a given numa_node in a PCI pool is
    # indicative of a PCI device being associated with that node
    #
    # The high level permutations rank algorithm is:
    # - permute remaining cells
    # - sort by non-PCI host nodes for non-PCI instances (if PCI not required)
    # - sort-by cell[0].id
    for itercells in sorted(
            itertools.permutations(
                list(iter(host_topology.cells)),
                len(instance_topology)),
            key=lambda x: (sum(y.id in [pool['numa_node']
                               for pool in pci_stats.pools] for y in x
                               if not pci_requests and pci_stats),
                           x[0].id)):
        host_cell_perm = list(itercells)
        cells = []
        for host_cell, instance_cell in zip(
                host_cell_perm, instance_topology.cells):
            try:
                cpuset_reserved = 0
                if (instance_topology.emulator_threads_isolated
                    and len(cells) == 0):
                    # For the case of isolate emulator threads, to
                    # make predictable where that CPU overhead is
                    # located we always configure it to be on host
                    # NUMA node associated to the guest NUMA node
                    # 0.
                    cpuset_reserved = 1
                got_cell = _numa_fit_instance_cell(
                    host_cell, instance_cell, limits, cpuset_reserved,
                    details=details)
            except exception.MemoryPageSizeNotSupported:
                # This exception will been raised if instance cell's
                # custom pagesize is not supported with host cell in
                # _numa_cell_supports_pagesize_request function.
                break
            if got_cell is None:
                break
            cells.append(got_cell)
        if len(cells) == len(host_cell_perm):
            if not pci_requests:
                return objects.InstanceNUMATopology(
                    cells=cells,
                    emulator_threads_policy=emulator_threads_policy)
            elif pci_stats is not None:
                if pci_stats.support_requests(pci_requests, cells,
                                              pci_strict=pci_strict):
                    return objects.InstanceNUMATopology(
                        cells=cells,
                        emulator_threads_policy=emulator_threads_policy)
                else:
                    msg = "PCI device not found or already in use"
                    details = utils.details_append(details, msg)


# WRS - pass through metrics
def numa_fit_instance_to_host(
        host_topology, instance_topology, limits=None,
        pci_requests=None, pci_stats=None,
        details=None,
        pci_strict=True):
    """Fit the instance topology onto the host topology.

    Given a host, instance topology, and (optional) limits, attempt to
    fit instance cells onto all permutations of host cells by calling
    the _fit_instance_cell method, and return a new InstanceNUMATopology
    with its cell ids set to host cell ids of the first successful
    permutation, or None.

    :param host_topology: objects.NUMATopology object to fit an
                          instance on
    :param instance_topology: objects.InstanceNUMATopology to be fitted
    :param limits: objects.NUMATopologyLimits that defines limits
    :param pci_requests: instance pci_requests
    :param pci_stats: pci_stats for the host

    :returns: objects.InstanceNUMATopology with its cell IDs set to host
              cell ids of the first successful permutation, or None
    """
    details = utils.details_initialize(details=details)
    if not (host_topology and instance_topology):
        LOG.debug("Require both a host and instance NUMA topology to "
                  "fit instance on host.")
        msg = 'Topology mismatch'
        details = utils.details_append(details, msg)
        return
    elif len(host_topology) < len(instance_topology):
        LOG.debug("There are not enough NUMA nodes on the system to schedule "
                  "the instance correctly. Required: %(required)s, actual: "
                  "%(actual)s",
                  {'required': len(instance_topology),
                   'actual': len(host_topology)})
        msg = ("Not enough free cores to schedule "
               "the instance. Required: %(required)s, actual: "
               "%(actual)s" %
               {'required': len(instance_topology),
                'actual': len(host_topology)})
        details = utils.details_append(details, msg)
        return

    # TODO(ndipanov): We may want to sort permutations differently
    # depending on whether we want packing/spreading over NUMA nodes
    # WRS: This code was re-factored.
    numa_topology = _numa_fit_instance_to_host(host_topology,
        instance_topology, limits, details, pci_requests, pci_stats,
        pci_strict=True)
    # WRS: If PCI strict allocation didn't succeed and pci_numa_affinity
    # is set to prefer, try again with best effort.
    if not numa_topology and pci_requests and not pci_strict:
        numa_topology = _numa_fit_instance_to_host(host_topology,
            instance_topology, limits, details, pci_requests, pci_stats,
            pci_strict=False)
    return numa_topology


def numa_get_reserved_huge_pages():
    """Returns reserved memory pages from host option.

    Based from the compute node option reserved_huge_pages, generate
    a well formatted list of dict which can be used to build a valid
    NUMATopology.

    :raises: exception.InvalidReservedMemoryPagesOption when
             reserved_huge_pages option is not correctly set.
    :returns: a list of dict ordered by NUMA node ids; keys of dict
              are pages size and values of the number reserved.
    """
    bucket = {}
    if CONF.reserved_huge_pages:
        try:
            bucket = collections.defaultdict(dict)
            for cfg in CONF.reserved_huge_pages:
                try:
                    pagesize = int(cfg['size'])
                except ValueError:
                    pagesize = strutils.string_to_bytes(
                        cfg['size'], return_int=True) / units.Ki
                bucket[int(cfg['node'])][pagesize] = int(cfg['count'])
        except (ValueError, TypeError, KeyError):
            raise exception.InvalidReservedMemoryPagesOption(
                conf=CONF.reserved_huge_pages)
    return bucket


def _numa_pagesize_usage_from_cell(hostcell, instancecell, sign):
    topo = []
    for pages in hostcell.mempages:
        if pages.size_kb == instancecell.pagesize:
            topo.append(objects.NUMAPagesTopology(
                size_kb=pages.size_kb,
                total=pages.total,
                used=max(0, pages.used +
                         instancecell.memory * units.Ki /
                         pages.size_kb * sign),
                reserved=pages.reserved if 'reserved' in pages else 0))
        else:
            topo.append(pages)
    return topo


def numa_usage_from_instances(host, instances, free=False, strict=True):
    """Get host topology usage.

    Sum the usage from all provided instances to report the overall
    host topology usage.

    :param host: objects.NUMATopology with usage information
    :param instances: list of objects.InstanceNUMATopology
    :param free: decrease, rather than increase, host usage

    :returns: objects.NUMATopology including usage information
    """
    if host is None:
        return

    instances = instances or []
    cells = []
    sign = -1 if free else 1
    for hostcell in host.cells:
        memory_usage = hostcell.memory_usage
        cpu_usage = hostcell.cpu_usage

        # L3 CAT Support
        l3_both_used = hostcell.l3_both_used
        l3_code_used = hostcell.l3_code_used
        l3_data_used = hostcell.l3_data_used

        # WRS: add shared_pcpu
        newcell = objects.NUMACell(
            id=hostcell.id, cpuset=hostcell.cpuset, memory=hostcell.memory,
            cpu_usage=0, memory_usage=0, mempages=hostcell.mempages,
            pinned_cpus=hostcell.pinned_cpus, siblings=hostcell.siblings,
            shared_pcpu=hostcell.shared_pcpu,
            l3_cdp=hostcell.l3_cdp,
            l3_size=hostcell.l3_size,
            l3_granularity=hostcell.l3_granularity,
            l3_both_used=0,
            l3_code_used=0,
            l3_data_used=0,
        )

        for instance in instances:
            for cellid, instancecell in enumerate(instance.cells):
                if instancecell.id == hostcell.id:
                    memory_usage = (
                            memory_usage + sign * instancecell.memory)
                    # If we can we want to use the number of unique pcpus since
                    # it will factor in scaled-down CPUs.  On initial creation
                    # however we may not have that info so use cpuset instead.
                    if instancecell.cpu_pinning is not None:
                        cpu_usage_diff = \
                            len(set(instancecell.cpu_pinning.values()))
                    else:
                        cpu_usage_diff = len(instancecell.cpuset)
                    if not instance.cpu_pinning_requested:
                        cpu_usage_diff /= float(CONF.cpu_allocation_ratio)
                    if (instancecell.cpu_thread_policy ==
                            fields.CPUThreadAllocationPolicy.ISOLATE and
                            hostcell.siblings):
                        cpu_usage_diff *= max(map(len, hostcell.siblings))
                    cpu_usage += sign * cpu_usage_diff

                    if (cellid == 0
                        and instance.emulator_threads_isolated):
                        # The emulator threads policy when defined
                        # with 'isolate' makes the instance to consume
                        # an additional pCPU as overhead. That pCPU is
                        # mapped on the host NUMA node related to the
                        # guest NUMA node 0.
                        cpu_usage += sign * len(instancecell.cpuset_reserved)

                    # L3 CAT support
                    if instance.cpu_pinning_requested:
                        if instancecell.l3_both_size is not None:
                            l3_both_used += \
                                sign * utils.roundup(instancecell.l3_both_size,
                                                     hostcell.l3_granularity)
                        if instancecell.l3_code_size is not None:
                            l3_code_used += \
                                sign * utils.roundup(instancecell.l3_code_size,
                                                     hostcell.l3_granularity)
                        if instancecell.l3_data_size is not None:
                            l3_data_used += \
                                sign * utils.roundup(instancecell.l3_data_size,
                                                     hostcell.l3_granularity)

                    if instancecell.pagesize and instancecell.pagesize > 0:
                        newcell.mempages = _numa_pagesize_usage_from_cell(
                            hostcell, instancecell, sign)
                    if instance.cpu_pinning_requested:
                        pinned_cpus = set(instancecell.cpu_pinning.values())
                        if instancecell.cpuset_reserved:
                            pinned_cpus |= instancecell.cpuset_reserved
                        if free:
                            if (instancecell.cpu_thread_policy ==
                                    fields.CPUThreadAllocationPolicy.ISOLATE):
                                e = newcell.unpin_cpus_with_siblings(
                                        pinned_cpus, strict=strict)
                            else:
                                e = newcell.unpin_cpus(pinned_cpus,
                                                       strict=strict)
                            if e:
                                LOG.error(
                                    'Cannot unpin:%(e)s (not pinned); '
                                    'requested:%(req)s',
                                    {'e': e, 'req': pinned_cpus})
                        else:
                            if (instancecell.cpu_thread_policy ==
                                    fields.CPUThreadAllocationPolicy.ISOLATE):
                                e = newcell.pin_cpus_with_siblings(
                                        pinned_cpus, strict=strict)
                            else:
                                e = newcell.pin_cpus(pinned_cpus,
                                                     strict=strict)
                            if e:
                                LOG.error(
                                    'Overlap pinning:%(e)s; '
                                    'requested:%(req)s',
                                    {'e': e, 'req': pinned_cpus})

        newcell.cpu_usage = max(0, cpu_usage)
        newcell.memory_usage = max(0, memory_usage)
        newcell.l3_both_used = max(0, l3_both_used)
        newcell.l3_code_used = max(0, l3_code_used)
        newcell.l3_data_used = max(0, l3_data_used)
        cells.append(newcell)

    return objects.NUMATopology(cells=cells)


# TODO(ndipanov): Remove when all code paths are using objects
def instance_topology_from_instance(instance):
    """Extract numa topology from myriad instance representations.

    Until the RPC version is bumped to 5.x, an instance may be
    represented as a dict, a db object, or an actual Instance object.
    Identify the type received and return either an instance of
    objects.InstanceNUMATopology if the instance's NUMA topology is
    available, else None.

    :param host: nova.objects.ComputeNode instance, or a db object or
                 dict

    :returns: An instance of objects.NUMATopology or None
    """
    if isinstance(instance, obj_instance.Instance):
        # NOTE (ndipanov): This may cause a lazy-load of the attribute
        instance_numa_topology = instance.numa_topology
    else:
        if 'numa_topology' in instance:
            instance_numa_topology = instance['numa_topology']
        elif 'uuid' in instance:
            try:
                instance_numa_topology = (
                    objects.InstanceNUMATopology.get_by_instance_uuid(
                            context.get_admin_context(), instance['uuid'])
                    )
            except exception.NumaTopologyNotFound:
                instance_numa_topology = None
        else:
            instance_numa_topology = None

    if instance_numa_topology:
        if isinstance(instance_numa_topology, six.string_types):
            instance_numa_topology = (
                objects.InstanceNUMATopology.obj_from_primitive(
                    jsonutils.loads(instance_numa_topology)))

        elif isinstance(instance_numa_topology, dict):
            # NOTE (ndipanov): A horrible hack so that we can use
            # this in the scheduler, since the
            # InstanceNUMATopology object is serialized raw using
            # the obj_base.obj_to_primitive, (which is buggy and
            # will give us a dict with a list of InstanceNUMACell
            # objects), and then passed to jsonutils.to_primitive,
            # which will make a dict out of those objects. All of
            # this is done by scheduler.utils.build_request_spec
            # called in the conductor.
            #
            # Remove when request_spec is a proper object itself!
            dict_cells = instance_numa_topology.get('cells')
            if dict_cells:
                cells = [objects.InstanceNUMACell(
                    id=cell['id'],
                    cpuset=set(cell['cpuset']),
                    memory=cell['memory'],
                    pagesize=cell.get('pagesize'),
                    cpu_topology=cell.get('cpu_topology'),
                    # WRS: add physnode
                    physnode=cell.get('physnode'),
                    # WRS: add shared_vcpu & shared_pcpu_for_vcpu
                    shared_vcpu=cell.get('shared_vcpu'),
                    shared_pcpu_for_vcpu=cell.get('shared_pcpu_for_vcpu'),
                    cpu_pinning=cell.get('cpu_pinning_raw'),
                    cpu_policy=cell.get('cpu_policy'),
                    cpu_thread_policy=cell.get('cpu_thread_policy'),
                    cpuset_reserved=cell.get('cpuset_reserved'),
                    # L3 CAT Support
                    l3_cpuset=set(cell.get('l3_cpuset') or []),
                    l3_both_size=cell.get('l3_both_size'),
                    l3_code_size=cell.get('l3_code_size'),
                    l3_data_size=cell.get('l3_data_size'))
                         for cell in dict_cells]
                emulator_threads_policy = instance_numa_topology.get(
                    'emulator_threads_policy')
                instance_numa_topology = objects.InstanceNUMATopology(
                    cells=cells,
                    emulator_threads_policy=emulator_threads_policy)

    return instance_numa_topology


# TODO(ndipanov): Remove when all code paths are using objects
def host_topology_and_format_from_host(host):
    """Extract numa topology from myriad host representations.

    Until the RPC version is bumped to 5.x, a host may be represented
    as a dict, a db object, an actual ComputeNode object, or an
    instance of HostState class. Identify the type received and return
    either an instance of objects.NUMATopology if host's NUMA topology
    is available, else None.

    :returns: A two-tuple. The first element is either an instance of
              objects.NUMATopology or None. The second element is a
              boolean set to True if topology was in JSON format.
    """
    was_json = False
    try:
        host_numa_topology = host.get('numa_topology')
    except AttributeError:
        host_numa_topology = host.numa_topology

    if host_numa_topology is not None and isinstance(
            host_numa_topology, six.string_types):
        was_json = True

        host_numa_topology = (objects.NUMATopology.obj_from_db_obj(
            host_numa_topology))

    return host_numa_topology, was_json


# TODO(ndipanov): Remove when all code paths are using objects
def get_host_numa_usage_from_instance(host, instance, free=False,
                                     never_serialize_result=False,
                                     strict=True):
    """Calculate new host NUMA usage from an instance's NUMA usage.

    Until the RPC version is bumped to 5.x, both host and instance
    representations may be provided in a variety of formats. Extract
    both host and instance numa topologies from provided
    representations, and use the latter to update the NUMA usage
    information of the former.

    :param host: nova.objects.ComputeNode instance, or a db object or
                 dict
    :param instance: nova.objects.Instance instance, or a db object or
                     dict
    :param free: if True the returned topology will have its usage
                 decreased instead
    :param never_serialize_result: if True result will always be an
                                   instance of objects.NUMATopology

    :returns: a objects.NUMATopology instance if never_serialize_result
              was True, else numa_usage in the format it was on the
              host
    """
    instance_numa_topology = instance_topology_from_instance(instance)
    if instance_numa_topology:
        instance_numa_topology = [instance_numa_topology]

    host_numa_topology, jsonify_result = host_topology_and_format_from_host(
            host)

    updated_numa_topology = (
        numa_usage_from_instances(
            host_numa_topology, instance_numa_topology,
            free=free, strict=strict))

    if updated_numa_topology is not None:
        if jsonify_result and not never_serialize_result:
            updated_numa_topology = updated_numa_topology._to_json()

    return updated_numa_topology


def instance_vcpu_to_pcpu(instance, vcpu):
    return instance.numa_topology.vcpu_to_pcpu(vcpu)


# WRS extension
def update_floating_affinity(host):
    """Update the CPU affinity for instances with non-dedicated CPUs.

    This is a helper function to update the CPU affinity of floating
    instances.  We assume that each floating instance is assigned to a cpuset,
    either global or per-host-NUMA-node.

    The kernel will not allow us to add a CPU to a subset before the global
    set, or remove a CPU from the global set before removing it from all
    subsets.  This means that we need to first remove CPUs from the subsets,
    then set the desired affinity in the global set, then set the desired
    affinity in the subsets.

    :param resources: the compute node resources
    :return: None
    """
    def get_cpuset_cpus(node):
        node_str = '' if node is None else '/node' + str(node)
        filename = CPUSET_BASE + node_str + '/cpuset.cpus'
        with open(filename, 'r') as f:
            cpus_string = f.read().rstrip('\n')
            return set(utils.range_to_list(cpus_string))

    def set_cpuset_cpus(node, cpus):
        node_str = '' if node is None else '/node' + str(node)
        filename = CPUSET_BASE + node_str + '/cpuset.cpus'
        # We want cpus separated by commas with no spaces
        cpus_string = ','.join(str(e) for e in cpus)
        try:
            with open(filename, 'w') as f:
                # Need to use this version, f.write() with an empty string
                # doesn't actually do a write() syscall.
                os.write(f.fileno(), cpus_string)
        except Exception as e:
            LOG.error('Unable to assign floating cpuset: %(cpus)r, '
                      'filename: %(f)s, error=%(e)s',
                      {'f': filename, 'cpus': cpus_string, 'e': e})

    host_numa_topology, jsonify_result = \
            host_topology_and_format_from_host(host)

    # Host doesn't report numa topology, can't continue.
    if host_numa_topology is None or not host_numa_topology.cells:
        return

    # Handle any CPU deletions from the subsets
    for cell in host_numa_topology.cells:
        cur_cpuset_cpus = get_cpuset_cpus(cell.id)
        new_cpuset_cpus = cur_cpuset_cpus.intersection(cell.free_cpus)
        set_cpuset_cpus(cell.id, new_cpuset_cpus)

    # Set the new global affinity
    floating_cpuset = set()
    for cell in host_numa_topology.cells:
        floating_cpuset.update(cell.free_cpus)
    set_cpuset_cpus(None, floating_cpuset)

    # Set the new affinity for all subsets
    for cell in host_numa_topology.cells:
        set_cpuset_cpus(cell.id, cell.free_cpus)


# WRS extension
def set_cpuset_tasks(node, tids):
    """Assign tasks represented by tids to the specified cpuset

    :param node: Either None or a NUMA node number
    :param tids: a set of Linux task IDs representing qemu vCPUs
    """
    node_str = '' if node is None else '/node' + str(node)
    filename = CPUSET_BASE + node_str + '/tasks'
    with open(filename, 'w') as f:
        for tid in tids:
            f.write(str(tid))
            # Need to flush because the kernel only takes one TID
            # per write() syscall.
            f.flush()
