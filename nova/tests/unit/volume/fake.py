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

import copy

from oslo_utils import uuidutils

from nova import exception
import nova.volume.cinder


class FakeVolumeAPI(nova.volume.cinder.API):
    def __init__(self):
        self.volumes = {}

    def initialize_connection(self, context, volume_id, connector):
        return {'connector': connector}

    def terminate_connection(self, context, volume_id, connector):
        pass

    def create(self, context, size, name, description, snapshot=None,
               image_id=None, volume_type=None, metadata=None,
               availability_zone=None):
        vol_id = uuidutils.generate_uuid()
        if snapshot is not None:
            snapshot_id = snapshot['id']
        else:
            snapshot_id = None
        data = dict(snapshot_id=snapshot_id,
                    volume_type=volume_type,
                    user_id=context.user_id,
                    project_id=context.project_id,
                    availability_zone=availability_zone,
                    metadata=metadata,
                    imageRef=image_id,
                    display_name=name,
                    display_description=description,
                    id=vol_id,
                    status='available',
                    attach_status='detached',
                    attachments={})
        self.volumes[vol_id] = data
        return copy.deepcopy(self.volumes[vol_id])

    def get(self, context, volume_id):
        volume = self.volumes.get(volume_id)
        if volume:
            return copy.deepcopy(volume)
        raise exception.VolumeNotFound(volume_id=volume_id)

    def attach(self, context, volume_id, instance_uuid, mountpoint, mode='rw'):
        volume = self.volumes.get(volume_id)
        if not volume:
            raise exception.VolumeNotFound(volume_id=volume_id)
        volume['attach_status'] = 'attached'
        volume['status'] = 'in-use'

    def detach(self, context, volume_id, instance_uuid=None,
               attachment_id=None):
        volume = self.volumes.get(volume_id)
        if not volume:
            raise exception.VolumeNotFound(volume_id=volume_id)
        volume['attach_status'] = 'detached'
        volume['status'] = 'available'


def stub_out_volume_api(test):
    test.stub_out('nova.volume.cinder.API', FakeVolumeAPI)
