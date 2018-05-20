#    Copyright 2014 Red Hat Inc.
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

from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import versionutils

from nova import db
from nova import exception
from nova.objects import base
from nova.objects import fields as obj_fields
from nova.virt import hardware


CONF = cfg.CONF


# TODO(berrange): Remove NovaObjectDictCompat
@base.NovaObjectRegistry.register
class InstanceNUMACell(base.NovaObject,
                       base.NovaObjectDictCompat):
    # Version 1.0: Initial version
    # Version 1.1: Add pagesize field
    # Version 1.2: Add cpu_pinning_raw and topology fields
    # Version 1.3: Add cpu_policy and cpu_thread_policy fields
    # Version 1.4: Add cpuset_reserved field
    #              WRS: Add physnode
    #              WRS: Add shared_vcpu and shared_pcpu_for_vcpu
    VERSION = '1.4'

    def obj_make_compatible(self, primitive, target_version):
        super(InstanceNUMACell, self).obj_make_compatible(primitive,
                                                        target_version)
        target_version = versionutils.convert_version_to_tuple(target_version)
        if target_version < (1, 4):
            primitive.pop('cpuset_reserved', None)

        if target_version < (1, 3):
            primitive.pop('cpu_policy', None)
            primitive.pop('cpu_thread_policy', None)
        # NOTE(jgauld): R4 to R5 upgrades, Pike upversion to 1.4. Drop L3
        #               related fields with R4/Newton.
        if target_version < (1, 4) or CONF.upgrade_levels.compute == 'newton':
            primitive.pop('l3_cpuset', None)
            primitive.pop('l3_both_size', None)
            primitive.pop('l3_code_size', None)
            primitive.pop('l3_data_size', None)

    fields = {
        'id': obj_fields.IntegerField(),
        'cpuset': obj_fields.SetOfIntegersField(),
        'memory': obj_fields.IntegerField(),
        'physnode': obj_fields.IntegerField(nullable=True),
        'pagesize': obj_fields.IntegerField(nullable=True),
        'cpu_topology': obj_fields.ObjectField('VirtCPUTopology',
                                               nullable=True),
        'cpu_pinning_raw': obj_fields.DictOfIntegersField(nullable=True),
        'shared_vcpu': obj_fields.IntegerField(nullable=True),
        'shared_pcpu_for_vcpu': obj_fields.IntegerField(nullable=True),
        'cpu_policy': obj_fields.CPUAllocationPolicyField(nullable=True),
        'cpu_thread_policy': obj_fields.CPUThreadAllocationPolicyField(
            nullable=True),
        # These physical CPUs are reserved for use by the hypervisor
        'cpuset_reserved': obj_fields.SetOfIntegersField(nullable=True),

        # L3 CAT
        'l3_cpuset': obj_fields.SetOfIntegersField(nullable=True),
        'l3_both_size': obj_fields.IntegerField(nullable=True),
        'l3_code_size': obj_fields.IntegerField(nullable=True),
        'l3_data_size': obj_fields.IntegerField(nullable=True),
    }

    cpu_pinning = obj_fields.DictProxyField('cpu_pinning_raw')

    def __init__(self, **kwargs):
        super(InstanceNUMACell, self).__init__(**kwargs)
        if 'pagesize' not in kwargs:
            self.pagesize = None
            self.obj_reset_changes(['pagesize'])
        if 'cpu_topology' not in kwargs:
            self.cpu_topology = None
            self.obj_reset_changes(['cpu_topology'])
        if 'cpu_pinning' not in kwargs:
            self.cpu_pinning = None
            self.obj_reset_changes(['cpu_pinning_raw'])
        if 'cpu_policy' not in kwargs:
            self.cpu_policy = None
            self.obj_reset_changes(['cpu_policy'])
        if 'cpu_thread_policy' not in kwargs:
            self.cpu_thread_policy = None
            self.obj_reset_changes(['cpu_thread_policy'])
        if 'cpuset_reserved' not in kwargs:
            self.cpuset_reserved = None
            self.obj_reset_changes(['cpuset_reserved'])
        if 'physnode' not in kwargs:
            self.physnode = None
            self.obj_reset_changes(['physnode'])
        if 'shared_vcpu' not in kwargs:
            self.shared_vcpu = None
            self.obj_reset_changes(['shared_vcpu'])
        if 'shared_pcpu_for_vcpu' not in kwargs:
            self.shared_pcpu_for_vcpu = None
            self.obj_reset_changes(['shared_pcpu_for_vcpu'])
        if 'l3_cpuset' not in kwargs:
            self.l3_cpuset = None
            self.obj_reset_changes(['l3_cpuset'])
        if 'l3_both_size' not in kwargs:
            self.l3_both_size = None
            self.obj_reset_changes(['l3_both_size'])
        if 'l3_code_size' not in kwargs:
            self.l3_code_size = None
            self.obj_reset_changes(['l3_code_size'])
        if 'l3_data_size' not in kwargs:
            self.l3_data_size = None
            self.obj_reset_changes(['l3_data_size'])

    def __len__(self):
        return len(self.cpuset)

    def _to_dict(self):
        # NOTE(sahid): Used as legacy, could be renamed in
        # _legacy_to_dict_ to the future to avoid confusing.
        return {'cpus': hardware.format_cpu_spec(self.cpuset,
                                                 allow_ranges=False),
                'mem': {'total': self.memory},
                'id': self.id,
                'pagesize': self.pagesize}

    @classmethod
    def _from_dict(cls, data_dict):
        # NOTE(sahid): Used as legacy, could be renamed in
        # _legacy_from_dict_ to the future to avoid confusing.
        cpuset = hardware.parse_cpu_spec(data_dict.get('cpus', ''))
        memory = data_dict.get('mem', {}).get('total', 0)
        cell_id = data_dict.get('id')
        pagesize = data_dict.get('pagesize')
        return cls(id=cell_id, cpuset=cpuset,
                   memory=memory, pagesize=pagesize)

    @property
    def siblings(self):
        cpu_list = sorted(list(self.cpuset))

        threads = 0
        if ('cpu_topology' in self) and self.cpu_topology:
            threads = self.cpu_topology.threads
        if threads == 1:
            threads = 0

        return list(map(set, zip(*[iter(cpu_list)] * threads)))

    @property
    def cpu_pinning_requested(self):
        return self.cpu_policy == obj_fields.CPUAllocationPolicy.DEDICATED

    def pin(self, vcpu, pcpu):
        if vcpu not in self.cpuset:
            return
        pinning_dict = self.cpu_pinning or {}
        pinning_dict[vcpu] = pcpu
        self.cpu_pinning = pinning_dict

    def pin_vcpus(self, *cpu_pairs):
        for vcpu, pcpu in cpu_pairs:
            self.pin(vcpu, pcpu)

    def clear_host_pinning(self):
        """Clear any data related to how this cell is pinned to the host.

        Needed for aborting claims as we do not want to keep stale data around.
        """
        self.id = -1
        self.cpu_pinning = {}
        return self

    @property
    def cachetune_requested(self):
        return (self.l3_cpuset is not None) and (len(self.l3_cpuset) > 0)

    # WRS extension
    @property
    def numa_pinning_requested(self):
        return self.physnode is not None

    # WRS: add a readable string representation
    def __str__(self):
        return '  {obj_name} (id: {id})\n' \
               '    cpuset: {cpuset}\n' \
               '    shared_vcpu: {shared_vcpu}\n' \
               '    shared_pcpu_for_vcpu: {shared_pcpu_for_vcpu}\n' \
               '    memory: {memory}\n' \
               '    physnode: {physnode}\n' \
               '    pagesize: {pagesize}\n' \
               '    cpu_topology: {cpu_topology}\n' \
               '    cpu_pinning: {cpu_pinning}\n' \
               '    siblings: {siblings}\n' \
               '    cpu_policy: {cpu_policy}\n' \
               '    cpu_thread_policy: {cpu_thread_policy}\n' \
               '    l3_cpuset: {l3_cpuset}\n' \
               '    l3_both_size: {l3_both_size}\n' \
               '    l3_code_size: {l3_code_size}\n' \
               '    l3_data_size: {l3_data_size}'.format(
            obj_name=self.obj_name(),
            id=self.id if ('id' in self) else None,
            cpuset=hardware.format_cpu_spec(
                self.cpuset, allow_ranges=True),
            shared_vcpu=self.shared_vcpu,
            shared_pcpu_for_vcpu=self.shared_pcpu_for_vcpu,
            memory=self.memory,
            physnode=self.physnode,
            pagesize=self.pagesize,
            cpu_topology=self.cpu_topology if (
                'cpu_topology' in self) else None,
            cpu_pinning=self.cpu_pinning,
            siblings=self.siblings,
            cpu_policy=self.cpu_policy,
            cpu_thread_policy=self.cpu_thread_policy,
            l3_cpuset=hardware.format_cpu_spec(
                self.l3_cpuset or [], allow_ranges=True),
            l3_both_size=self.l3_both_size,
            l3_code_size=self.l3_code_size,
            l3_data_size=self.l3_data_size,
        )

    # WRS: add a readable representation, without newlines
    def __repr__(self):
        return '{obj_name} (id: {id}) ' \
               'cpuset: {cpuset} ' \
               'shared_vcpu: {shared_vcpu} ' \
               'shared_pcpu_for_vcpu: {shared_pcpu_for_vcpu} ' \
               'memory: {memory} ' \
               'physnode: {physnode} ' \
               'pagesize: {pagesize} ' \
               'cpu_topology: {cpu_topology} ' \
               'cpu_pinning: {cpu_pinning} ' \
               'siblings: {siblings} ' \
               'cpu_policy: {cpu_policy} ' \
               'cpu_thread_policy: {cpu_thread_policy} ' \
               'l3_cpuset: {l3_cpuset} ' \
               'l3_both_size: {l3_both_size} ' \
               'l3_code_size: {l3_code_size} ' \
               'l3_data_size: {l3_data_size}'.format(
            obj_name=self.obj_name(),
            id=self.id if ('id' in self) else None,
            cpuset=hardware.format_cpu_spec(
                self.cpuset, allow_ranges=True),
            shared_vcpu=self.shared_vcpu,
            shared_pcpu_for_vcpu=self.shared_pcpu_for_vcpu,
            memory=self.memory,
            physnode=self.physnode,
            pagesize=self.pagesize,
            cpu_topology=self.cpu_topology if (
                'cpu_topology' in self) else None,
            cpu_pinning=self.cpu_pinning,
            siblings=self.siblings,
            cpu_policy=self.cpu_policy,
            cpu_thread_policy=self.cpu_thread_policy,
            l3_cpuset=hardware.format_cpu_spec(
                self.l3_cpuset or [], allow_ranges=True),
            l3_both_size=self.l3_both_size,
            l3_code_size=self.l3_code_size,
            l3_data_size=self.l3_data_size,
        )


# TODO(berrange): Remove NovaObjectDictCompat
@base.NovaObjectRegistry.register
class InstanceNUMATopology(base.NovaObject,
                           base.NovaObjectDictCompat):
    # Version 1.0: Initial version
    # Version 1.1: Takes into account pagesize
    # Version 1.2: InstanceNUMACell 1.2
    # Version 1.3: Add emulator threads policy
    VERSION = '1.3'

    def obj_make_compatible(self, primitive, target_version):
        super(InstanceNUMATopology, self).obj_make_compatible(primitive,
                                                        target_version)
        target_version = versionutils.convert_version_to_tuple(target_version)
        if target_version < (1, 3):
            primitive.pop('emulator_threads_policy', None)

    fields = {
        # NOTE(danms): The 'id' field is no longer used and should be
        # removed in the future when convenient
        'id': obj_fields.IntegerField(),
        'instance_uuid': obj_fields.UUIDField(),
        'cells': obj_fields.ListOfObjectsField('InstanceNUMACell'),
        'emulator_threads_policy': (
            obj_fields.CPUEmulatorThreadsPolicyField(nullable=True)),
        }

    @classmethod
    def obj_from_primitive(cls, primitive, context=None):
        if 'nova_object.name' in primitive:
            obj_topology = super(InstanceNUMATopology, cls).obj_from_primitive(
                primitive, context=None)
        else:
            # NOTE(sahid): This compatibility code needs to stay until we can
            # guarantee that there are no cases of the old format stored in
            # the database (or forever, if we can never guarantee that).
            obj_topology = InstanceNUMATopology._from_dict(primitive)
            obj_topology.id = 0
        return obj_topology

    @classmethod
    def obj_from_db_obj(cls, instance_uuid, db_obj):
        primitive = jsonutils.loads(db_obj)
        obj_topology = cls.obj_from_primitive(primitive)

        if 'nova_object.name' not in db_obj:
            obj_topology.instance_uuid = instance_uuid
            # No benefit to store a list of changed fields
            obj_topology.obj_reset_changes()

        return obj_topology

    # TODO(ndipanov) Remove this method on the major version bump to 2.0
    @base.remotable
    def create(self):
        values = {'numa_topology': self._to_json()}
        db.instance_extra_update_by_uuid(self._context, self.instance_uuid,
                                         values)
        self.obj_reset_changes()

    @base.remotable_classmethod
    def get_by_instance_uuid(cls, context, instance_uuid):
        db_extra = db.instance_extra_get_by_instance_uuid(
                context, instance_uuid, columns=['numa_topology'])
        if not db_extra:
            raise exception.NumaTopologyNotFound(instance_uuid=instance_uuid)

        if db_extra['numa_topology'] is None:
            return None

        return cls.obj_from_db_obj(instance_uuid, db_extra['numa_topology'])

    def _to_json(self):
        return jsonutils.dumps(self.obj_to_primitive())

    def __len__(self):
        """Defined so that boolean testing works the same as for lists."""
        return len(self.cells)

    def _to_dict(self):
        # NOTE(sahid): Used as legacy, could be renamed in _legacy_to_dict_
        # in the future to avoid confusing.
        return {'cells': [cell._to_dict() for cell in self.cells]}

    @classmethod
    def _from_dict(cls, data_dict):
        # NOTE(sahid): Used as legacy, could be renamed in _legacy_from_dict_
        # in the future to avoid confusing.
        return cls(cells=[
            InstanceNUMACell._from_dict(cell_dict)
            for cell_dict in data_dict.get('cells', [])])

    @property
    def cpu_pinning_requested(self):
        return all(cell.cpu_pinning_requested for cell in self.cells)

    def clear_host_pinning(self):
        """Clear any data related to how instance is pinned to the host.

        Needed for aborting claims as we do not want to keep stale data around.
        """
        for cell in self.cells:
            cell.clear_host_pinning()
        return self

    def __str__(self):
        topology_str = '{obj_name}:'.format(obj_name=self.obj_name())
        for cell in self.cells:
            topology_str += '\n' + str(cell)
        return topology_str

    def __repr__(self):
        topology_str = '{obj_name}: '.format(obj_name=self.obj_name())
        topology_str += ', '.join(repr(cell) for cell in self.cells)
        return topology_str

    @property
    def emulator_threads_isolated(self):
        """Determines whether emulator threads should be isolated"""
        return (self.obj_attr_is_set('emulator_threads_policy')
                and (self.emulator_threads_policy
                     == obj_fields.CPUEmulatorThreadsPolicy.ISOLATE))

    @property
    def numa_pinning_requested(self):
        return all(cell.numa_pinning_requested for cell in self.cells)

    def vcpu_to_pcpu(self, vcpu):
        for cell in self.cells:
            if vcpu in cell.cpu_pinning.keys():
                return cell, cell.cpu_pinning[vcpu]
            if vcpu == cell.shared_vcpu:
                return cell, cell.shared_pcpu_for_vcpu
        raise KeyError('Unable to find pCPU for vCPU %d' % vcpu)

    @property
    def offline_cpus(self):
        offline_cpuset = set()
        if not self.cpu_pinning_requested:
            return offline_cpuset
        # The offline vCPUs will be pinned the same as vCPU0
        # or the shared vcpu index if it is assigned
        for cell in self.cells:
            online_index = 0
            if cell.shared_vcpu is not None:
                online_index = cell.shared_vcpu
            vcpu0_cell, vcpu0_phys = self.vcpu_to_pcpu(online_index)
            for vcpu in cell.cpuset:
                if (vcpu != online_index
                    and cell.cpu_pinning[vcpu] == vcpu0_phys):
                    offline_cpuset |= {vcpu}
        return offline_cpuset

    def set_cpus_offline(self, offline_cpus):
        if not self.cpu_pinning_requested:
            return
        # The offline vCPUs will be pinned the same as vCPU0
        for cell in self.cells:
            online_index = 0
            if cell.shared_vcpu is not None:
                online_index = cell.shared_vcpu
            vcpu0_cell, vcpu0_phys = self.vcpu_to_pcpu(online_index)
            for vcpu in cell.cpuset:
                if vcpu in offline_cpus:
                    cell.pin(vcpu, vcpu0_phys)
