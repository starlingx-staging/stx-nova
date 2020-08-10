#    Copyright 2013 Red Hat Inc.
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

from oslo_db import api as oslo_db_api
from oslo_db.sqlalchemy import update_match
from oslo_log import log as logging
from oslo_utils import uuidutils
from oslo_utils import versionutils

from nova import block_device
from nova.db import api as db
from nova.db.sqlalchemy import api as db_api
from nova.db.sqlalchemy import models as db_models
from nova import exception
from nova.i18n import _
from nova import objects
from nova.objects import base
from nova.objects import fields


LOG = logging.getLogger(__name__)


_BLOCK_DEVICE_OPTIONAL_JOINED_FIELD = ['instance']
BLOCK_DEVICE_OPTIONAL_ATTRS = _BLOCK_DEVICE_OPTIONAL_JOINED_FIELD


def _expected_cols(expected_attrs):
    return [attr for attr in expected_attrs
                 if attr in _BLOCK_DEVICE_OPTIONAL_JOINED_FIELD]


# TODO(berrange): Remove NovaObjectDictCompat
@base.NovaObjectRegistry.register
class BlockDeviceMapping(base.NovaPersistentObject, base.NovaObject,
                         base.NovaObjectDictCompat):
    # Version 1.0: Initial version
    # Version 1.1: Add instance_uuid to get_by_volume_id method
    # Version 1.2: Instance version 1.14
    # Version 1.3: Instance version 1.15
    # Version 1.4: Instance version 1.16
    # Version 1.5: Instance version 1.17
    # Version 1.6: Instance version 1.18
    # Version 1.7: Add update_or_create method
    # Version 1.8: Instance version 1.19
    # Version 1.9: Instance version 1.20
    # Version 1.10: Changed source_type field to BlockDeviceSourceTypeField.
    # Version 1.11: Changed destination_type field to
    #               BlockDeviceDestinationTypeField.
    # Version 1.12: Changed device_type field to BlockDeviceTypeField.
    # Version 1.13: Instance version 1.21
    # Version 1.14: Instance version 1.22
    # Version 1.15: Instance version 1.23
    # Version 1.16: Deprecate get_by_volume_id(), add
    #               get_by_volume() and get_by_volume_and_instance()
    # Version 1.17: Added tag field
    # Version 1.18: Added attachment_id
    # Version 1.19: Added uuid
    # Version 1.20: Added volume_type
    VERSION = '1.20'

    fields = {
        'id': fields.IntegerField(),
        'uuid': fields.UUIDField(),
        'instance_uuid': fields.UUIDField(),
        'instance': fields.ObjectField('Instance', nullable=True),
        'source_type': fields.BlockDeviceSourceTypeField(nullable=True),
        'destination_type': fields.BlockDeviceDestinationTypeField(
                                nullable=True),
        'guest_format': fields.StringField(nullable=True),
        'device_type': fields.BlockDeviceTypeField(nullable=True),
        'disk_bus': fields.StringField(nullable=True),
        'boot_index': fields.IntegerField(nullable=True),
        'device_name': fields.StringField(nullable=True),
        'delete_on_termination': fields.BooleanField(default=False),
        'snapshot_id': fields.StringField(nullable=True),
        'volume_id': fields.StringField(nullable=True),
        'volume_size': fields.IntegerField(nullable=True),
        'image_id': fields.StringField(nullable=True),
        'no_device': fields.BooleanField(default=False),
        'connection_info': fields.SensitiveStringField(nullable=True),
        'tag': fields.StringField(nullable=True),
        'attachment_id': fields.UUIDField(nullable=True),
        # volume_type field can be a volume type name or ID(UUID).
        'volume_type': fields.StringField(nullable=True),
    }

    def obj_make_compatible(self, primitive, target_version):
        target_version = versionutils.convert_version_to_tuple(target_version)
        if target_version < (1, 20) and 'volume_type' in primitive:
            del primitive['volume_type']
        if target_version < (1, 19) and 'uuid' in primitive:
            del primitive['uuid']
        if target_version < (1, 18) and 'attachment_id' in primitive:
            del primitive['attachment_id']
        if target_version < (1, 17) and 'tag' in primitive:
            del primitive['tag']

    @classmethod
    def populate_uuids(cls, context, count):
        @db_api.pick_context_manager_reader
        def get_bdms_no_uuid(context):
            return context.session.query(db_models.BlockDeviceMapping).\
                    filter_by(uuid=None).limit(count).all()

        db_bdms = get_bdms_no_uuid(context)

        done = 0
        for db_bdm in db_bdms:
            cls._create_uuid(context, db_bdm['id'])
            done += 1

        return done, done

    @staticmethod
    @oslo_db_api.wrap_db_retry(max_retries=1, retry_on_deadlock=True)
    def _create_uuid(context, bdm_id):
        # NOTE(mdbooth): This method is only required until uuid is made
        # non-nullable in a future release.

        # NOTE(mdbooth): We wrap this method in a retry loop because it can
        # fail (safely) on multi-main galera if concurrent updates happen on
        # different mains. It will never fail on single-main. We can only
        # ever need one retry.

        uuid = uuidutils.generate_uuid()
        values = {'uuid': uuid}
        compare = db_models.BlockDeviceMapping(id=bdm_id, uuid=None)

        # NOTE(mdbooth): We explicitly use an independent transaction context
        # here so as not to fail if:
        # 1. We retry.
        # 2. We're in a read transaction.This is an edge case of what's
        #    normally a read operation. Forcing everything (transitively)
        #    which reads a BDM to be in a write transaction for a narrow
        #    temporary edge case is undesirable.
        tctxt = db_api.get_context_manager(context).writer.independent
        with tctxt.using(context):
            query = context.session.query(db_models.BlockDeviceMapping).\
                        filter_by(id=bdm_id)

            try:
                query.update_on_match(compare, 'id', values)
            except update_match.NoRowsMatched:
                # We can only get here if we raced, and another writer already
                # gave this bdm a uuid
                result = query.one()
                uuid = result['uuid']
                assert(uuid is not None)

        return uuid

    @classmethod
    def _from_db_object(cls, context, block_device_obj,
                        db_block_device, expected_attrs=None):
        if expected_attrs is None:
            expected_attrs = []
        for key in block_device_obj.fields:
            if key in BLOCK_DEVICE_OPTIONAL_ATTRS:
                continue
            if key == 'uuid' and not db_block_device.get(key):
                # NOTE(danms): While the records could be nullable,
                # generate a UUID on read since the object requires it
                bdm_id = db_block_device['id']
                db_block_device[key] = cls._create_uuid(context, bdm_id)
            block_device_obj[key] = db_block_device[key]
        if 'instance' in expected_attrs:
            my_inst = objects.Instance(context)
            my_inst._from_db_object(context, my_inst,
                                    db_block_device['instance'])
            block_device_obj.instance = my_inst

        block_device_obj._context = context
        block_device_obj.obj_reset_changes()
        return block_device_obj

    def _create(self, context, update_or_create=False):
        """Create the block device record in the database.

        In case the id field is set on the object, and if the instance is set
        raise an ObjectActionError. Resets all the changes on the object.

        Returns None

        :param context: security context used for database calls
        :param update_or_create: consider existing block devices for the
                instance based on the device name and swap, and only update
                the ones that match. Normally only used when creating the
                instance for the first time.
        """
        if self.obj_attr_is_set('id'):
            raise exception.ObjectActionError(action='create',
                                              reason='already created')
        updates = self.obj_get_changes()
        if 'instance' in updates:
            raise exception.ObjectActionError(action='create',
                                              reason='instance assigned')

        if update_or_create:
            db_bdm = db.block_device_mapping_update_or_create(
                    context, updates, legacy=False)
        else:
            db_bdm = db.block_device_mapping_create(
                    context, updates, legacy=False)

        self._from_db_object(context, self, db_bdm)

    @base.remotable
    def create(self):
        self._create(self._context)

    @base.remotable
    def update_or_create(self):
        self._create(self._context, update_or_create=True)

    @base.remotable
    def destroy(self):
        if not self.obj_attr_is_set('id'):
            raise exception.ObjectActionError(action='destroy',
                                              reason='already destroyed')
        db.block_device_mapping_destroy(self._context, self.id)
        delattr(self, base.get_attrname('id'))

    @base.remotable
    def save(self):
        updates = self.obj_get_changes()
        if 'instance' in updates:
            raise exception.ObjectActionError(action='save',
                                              reason='instance changed')
        updates.pop('id', None)
        updated = db.block_device_mapping_update(self._context, self.id,
                                                 updates, legacy=False)
        if not updated:
            raise exception.BDMNotFound(id=self.id)
        self._from_db_object(self._context, self, updated)

    # NOTE(danms): This method is deprecated and will be removed in
    # v2.0 of the object
    @base.remotable_classmethod
    def get_by_volume_id(cls, context, volume_id,
                         instance_uuid=None, expected_attrs=None):
        if expected_attrs is None:
            expected_attrs = []
        db_bdms = db.block_device_mapping_get_all_by_volume_id(
                context, volume_id, _expected_cols(expected_attrs))
        if not db_bdms:
            raise exception.VolumeBDMNotFound(volume_id=volume_id)
        if len(db_bdms) > 1:
            LOG.warning('Legacy get_by_volume_id() call found multiple '
                        'BDMs for volume %(volume)s',
                        {'volume': volume_id})
        db_bdm = db_bdms[0]
        # NOTE (ndipanov): Move this to the db layer into a
        # get_by_instance_and_volume_id method
        if instance_uuid and instance_uuid != db_bdm['instance_uuid']:
            raise exception.InvalidVolume(
                    reason=_("Volume does not belong to the "
                             "requested instance."))
        return cls._from_db_object(context, cls(), db_bdm,
                                   expected_attrs=expected_attrs)

    @base.remotable_classmethod
    def get_by_volume_and_instance(cls, context, volume_id, instance_uuid,
                                   expected_attrs=None):
        if expected_attrs is None:
            expected_attrs = []
        db_bdm = db.block_device_mapping_get_by_instance_and_volume_id(
            context, volume_id, instance_uuid,
            _expected_cols(expected_attrs))
        if not db_bdm:
            raise exception.VolumeBDMNotFound(volume_id=volume_id)
        return cls._from_db_object(context, cls(), db_bdm,
                                   expected_attrs=expected_attrs)

    @base.remotable_classmethod
    def get_by_volume(cls, context, volume_id, expected_attrs=None):
        if expected_attrs is None:
            expected_attrs = []
        db_bdms = db.block_device_mapping_get_all_by_volume_id(
                context, volume_id, _expected_cols(expected_attrs))
        if not db_bdms:
            raise exception.VolumeBDMNotFound(volume_id=volume_id)
        if len(db_bdms) > 1:
            raise exception.VolumeBDMIsMultiAttach(volume_id=volume_id)
        return cls._from_db_object(context, cls(), db_bdms[0],
                                   expected_attrs=expected_attrs)

    @property
    def is_root(self):
        return self.boot_index == 0

    @property
    def is_volume(self):
        return (self.destination_type ==
                    fields.BlockDeviceDestinationType.VOLUME)

    @property
    def is_image(self):
        return self.source_type == fields.BlockDeviceSourceType.IMAGE

    def get_image_mapping(self):
        return block_device.BlockDeviceDict(self).get_image_mapping()

    def obj_load_attr(self, attrname):
        if attrname not in BLOCK_DEVICE_OPTIONAL_ATTRS:
            raise exception.ObjectActionError(
                action='obj_load_attr',
                reason='attribute %s not lazy-loadable' % attrname)
        if not self._context:
            raise exception.OrphanedObjectError(method='obj_load_attr',
                                                objtype=self.obj_name())

        LOG.debug("Lazy-loading '%(attr)s' on %(name)s using uuid %(uuid)s",
                  {'attr': attrname,
                   'name': self.obj_name(),
                   'uuid': self.instance_uuid,
                   })
        self.instance = objects.Instance.get_by_uuid(self._context,
                                                     self.instance_uuid)
        self.obj_reset_changes(fields=['instance'])


@base.NovaObjectRegistry.register
class BlockDeviceMappingList(base.ObjectListBase, base.NovaObject):
    # Version 1.0: Initial version
    # Version 1.1: BlockDeviceMapping <= version 1.1
    # Version 1.2: Added use_subordinate to get_by_instance_uuid
    # Version 1.3: BlockDeviceMapping <= version 1.2
    # Version 1.4: BlockDeviceMapping <= version 1.3
    # Version 1.5: BlockDeviceMapping <= version 1.4
    # Version 1.6: BlockDeviceMapping <= version 1.5
    # Version 1.7: BlockDeviceMapping <= version 1.6
    # Version 1.8: BlockDeviceMapping <= version 1.7
    # Version 1.9: BlockDeviceMapping <= version 1.8
    # Version 1.10: BlockDeviceMapping <= version 1.9
    # Version 1.11: BlockDeviceMapping <= version 1.10
    # Version 1.12: BlockDeviceMapping <= version 1.11
    # Version 1.13: BlockDeviceMapping <= version 1.12
    # Version 1.14: BlockDeviceMapping <= version 1.13
    # Version 1.15: BlockDeviceMapping <= version 1.14
    # Version 1.16: BlockDeviceMapping <= version 1.15
    # Version 1.17: Add get_by_instance_uuids()
    VERSION = '1.17'

    fields = {
        'objects': fields.ListOfObjectsField('BlockDeviceMapping'),
    }

    @property
    def instance_uuids(self):
        return set(
            bdm.instance_uuid for bdm in self
            if bdm.obj_attr_is_set('instance_uuid')
        )

    @classmethod
    def bdms_by_instance_uuid(cls, context, instance_uuids):
        bdms = cls.get_by_instance_uuids(context, instance_uuids)
        return base.obj_make_dict_of_lists(
                context, cls, bdms, 'instance_uuid')

    @staticmethod
    @db.select_db_reader_mode
    def _db_block_device_mapping_get_all_by_instance_uuids(
            context, instance_uuids, use_subordinate=False):
        return db.block_device_mapping_get_all_by_instance_uuids(
                context, instance_uuids)

    @base.remotable_classmethod
    def get_by_instance_uuids(cls, context, instance_uuids, use_subordinate=False):
        db_bdms = cls._db_block_device_mapping_get_all_by_instance_uuids(
            context, instance_uuids, use_subordinate=use_subordinate)
        return base.obj_make_list(
                context, cls(), objects.BlockDeviceMapping, db_bdms or [])

    @staticmethod
    @db.select_db_reader_mode
    def _db_block_device_mapping_get_all_by_instance(
            context, instance_uuid, use_subordinate=False):
        return db.block_device_mapping_get_all_by_instance(
            context, instance_uuid)

    @base.remotable_classmethod
    def get_by_instance_uuid(cls, context, instance_uuid, use_subordinate=False):
        db_bdms = cls._db_block_device_mapping_get_all_by_instance(
            context, instance_uuid, use_subordinate=use_subordinate)
        return base.obj_make_list(
                context, cls(), objects.BlockDeviceMapping, db_bdms or [])

    def root_bdm(self):
        """It only makes sense to call this method when the
        BlockDeviceMappingList contains BlockDeviceMappings from
        exactly one instance rather than BlockDeviceMappings from
        multiple instances.

        For example, you should not call this method from a
        BlockDeviceMappingList created by get_by_instance_uuids(),
        but you may call this method from a BlockDeviceMappingList
        created by get_by_instance_uuid().
        """

        if len(self.instance_uuids) > 1:
            raise exception.UndefinedRootBDM()
        try:
            return next(bdm_obj for bdm_obj in self if bdm_obj.is_root)
        except StopIteration:
            return


def block_device_make_list(context, db_list, **extra_args):
    return base.obj_make_list(context,
                              objects.BlockDeviceMappingList(context),
                              objects.BlockDeviceMapping, db_list,
                              **extra_args)


def block_device_make_list_from_dicts(context, bdm_dicts_list):
    bdm_objects = [objects.BlockDeviceMapping(context=context, **bdm)
                   for bdm in bdm_dicts_list]
    return BlockDeviceMappingList(objects=bdm_objects)
