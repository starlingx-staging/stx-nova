#    Copyright 2010 United States Government as represented by the
#    Administrator of the National Aeronautics and Space Administration.
#    All Rights Reserved.
#    Copyright (c) 2010 Citrix Systems, Inc.
#    Copyright (c) 2011 Piston Cloud Computing, Inc
#    Copyright (c) 2011 OpenStack Foundation
#    (c) Copyright 2013 Hewlett-Packard Development Company, L.P.
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

import os.path
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import units
import six

import nova.conf
from nova import exception
from nova.i18n import _
from nova.virt.libvirt import utils

CONF = nova.conf.CONF
LOG = logging.getLogger(__name__)


def create_thinpool_if_needed(vg):
    if CONF.libvirt.thin_logical_volumes is False:
        return

    poolname = vg + CONF.libvirt.thinpool_suffix
    # Do we need to worry about resizing the VG and thinpool?
    if poolname in list_volumes(vg):
        return
    # Create thinpool.  Leave 5% of free space for metadata.
    size = _get_volume_group_info(vg)['free'] * 0.95
    # Round down to the nearest GiB like cinder.  This also means we don't
    # need to worry about being a multiple of the chunk size.
    size = int(size) >> 30 << 30
    create_volume(vg, poolname, size)


def thin_copy_volume(lv_src, lv_dest, vg, size):
    """Copies a thin-provisioned volume.

    Because we know we're dealing with thin-provisioned volumes, we can just
    do a snapshot.  (As long as there's enough free space.)

    :param lv_src: source thin volume
    :param lv_dest: dest thin volume
    :param vg: volume group
    :param size: size of the volume being copied

    This duplicates some code from create_duplicate_volume() and from
    create_volume(), the alternative would be to mangle create_volume() to
    also handle snapshots and then tweak a bunch of unit tests.
    """
    thinpool = vg + CONF.libvirt.thinpool_suffix
    vg_info = _get_thinpool_info(vg, thinpool)
    free_space = vg_info['free']
    if size > free_space:
        raise RuntimeError(_('Insufficient Space on Volume Group %(vg)s.'
                             ' Only %(free_space)db available,'
                             ' but %(size)db required'
                             ' by thin volume %(lv)s.') %
                           {'vg': vg,
                            'free_space': free_space,
                            'size': size,
                            'lv': lv_dest})
    cmd = ('lvcreate', '-s', '-kn', '-n', lv_dest, '%s/%s' % (vg, lv_src))
    utils.execute(*cmd, run_as_root=True, attempts=3)


def create_volume(vg, lv, size, sparse=False):
    """Create LVM image.

    Creates a LVM image with given size.

    :param vg: existing volume group which should hold this image
    :param lv: name for this volume (logical volume)
    :size: size of image in bytes
    :sparse: create sparse logical volume
    """

    # Figure out thinpool name and type of volume to create.
    thinpool = vg + CONF.libvirt.thinpool_suffix
    if CONF.libvirt.thin_logical_volumes:
        if lv == thinpool:
            vtype = 'thinpool'
        else:
            vtype = 'thin'
    else:
        vtype = 'default'

    # Can't call get_volume_group_info() because we want to do special
    # handling when creating the thinpool itself.
    if vtype == 'thin':
        vg_info = _get_thinpool_info(vg, thinpool)
    else:
        vg_info = _get_volume_group_info(vg)
    free_space = vg_info['free']

    def check_size(vg, lv, size):
        if size > free_space:
            raise RuntimeError(_('Insufficient Space on Volume Group %(vg)s.'
                                 ' Only %(free_space)db available,'
                                 ' but %(size)d bytes required'
                                 ' by volume %(lv)s.') %
                               {'vg': vg,
                                'free_space': free_space,
                                'size': size,
                                'lv': lv})

    if sparse:
        preallocated_space = 64 * units.Mi
        check_size(vg, lv, preallocated_space)
        if free_space < size:
            LOG.warning('Volume group %(vg)s will not be able'
                        ' to hold sparse volume %(lv)s.'
                        ' Virtual volume size is %(size)d bytes,'
                        ' but free space on volume group is'
                        ' only %(free_space)db.',
                        {'vg': vg,
                         'free_space': free_space,
                         'size': size,
                         'lv': lv})

        cmd = ('lvcreate', '-L', '%db' % preallocated_space,
                '--virtualsize', '%db' % size, '-n', lv, vg)
    else:
        check_size(vg, lv, size)
        if vtype == 'default':
            cmd = ('lvcreate', '-L', '%db' % size, '-n', lv, vg)
        elif vtype == 'thinpool':
            cmd = ('lvcreate', '-L', '%db' % size, '-T', '%s/%s' % (vg, lv))
        elif vtype == 'thin':
            cmd = ('lvcreate', '-V', '%db' % size, '-T',
                   '%s/%s' % (vg, thinpool), '-n', lv)

    utils.execute(*cmd, run_as_root=True, attempts=3)


def _get_thinpool_info(vg, thinpool):
    """Return free/used/total space info for a thinpool in the specified vg

    :param vg: volume group name
    :param thinpool: thin pool name
    :returns: A dict containing:
             :total: How big the filesystem is (in bytes)
             :free: How much space is free (in bytes)
             :used: How much space is used (in bytes)
    """
    thinpool_used = 0
    thinpool_size = 0
    out, err = utils.execute('vgs', '--noheadings', '--nosuffix',
                       '--separator', '|',
                       '--units', 'b', '-o', 'lv_name,lv_size,pool_lv', vg,
                       run_as_root=True)
    for line in out.splitlines():
        lvinfo = line.split('|')
        lv_name = lvinfo[0].strip()
        lv_size = int(lvinfo[1])
        lv_pool = lvinfo[2]
        if lv_name == thinpool:
            thinpool_size = lv_size
        elif lv_pool == thinpool:
            # Account for total size of all volumes in the thin pool.
            thinpool_used += lv_size
    return {'total': thinpool_size,
            'free': thinpool_size - thinpool_used,
            'used': thinpool_used}


def _get_volume_group_info(vg):
    """Return free/used/total space info for a volume group in bytes

    :param vg: volume group name
    :returns: A dict containing:
             :total: How big the filesystem is (in bytes)
             :free: How much space is free (in bytes)
             :used: How much space is used (in bytes)
    """

    out, err = utils.execute('vgs', '--noheadings', '--nosuffix',
                       '--separator', '|',
                       '--units', 'b', '-o', 'vg_size,vg_free', vg,
                       run_as_root=True)

    info = out.split('|')
    if len(info) != 2:
        raise RuntimeError(_("vg %s must be LVM volume group") % vg)

    return {'total': int(info[0]),
            'free': int(info[1]),
            'used': int(info[0]) - int(info[1])}


def get_volume_group_info(vg):
    """Return free/used/total space info in bytes

    If thin provisioning is enabled then return data for the thin pool,
    otherwise return data for the volume group.
    """
    if CONF.libvirt.thin_logical_volumes:
        thinpool = vg + CONF.libvirt.thinpool_suffix
        return _get_thinpool_info(vg, thinpool)
    else:
        return _get_volume_group_info(vg)


def list_volumes(vg):
    """List logical volumes paths for given volume group.

    :param vg: volume group name
    :returns: Return a logical volume list for given volume group
            : Data format example
            : ['volume-aaa', 'volume-bbb', 'volume-ccc']
    """
    out, err = utils.execute('lvs', '--noheadings', '-o', 'lv_name', vg,
                             run_as_root=True)

    return [line.strip() for line in out.splitlines()]


def volume_info(path):
    """Get logical volume info.

    :param path: logical volume path
    :returns: Return a dict object including info of given logical volume
            : Data format example
            : {'#Seg': '1', 'Move': '', 'Log': '', 'Meta%': '', 'Min': '-1',
            : ...
            : 'Free': '9983', 'LV': 'volume-aaa', 'Host': 'xyz.com',
            : 'Active': 'active', 'Path': '/dev/vg/volume-aaa', '#LV': '3',
            : 'Maj': '-1', 'VSize': '50.00g', 'VFree': '39.00g', 'Pool': '',
            : 'VG Tags': '', 'KMaj': '253', 'Convert': '', 'LProfile': '',
            : '#Ext': '12799', 'Attr': '-wi-a-----', 'VG': 'vg',
            : ...
            : 'LSize': '1.00g', '#PV': '1', '#VMdaCps': 'unmanaged'}
    """
    out, err = utils.execute('lvs', '-o', 'vg_all,lv_all',
                             '--separator', '|', path, run_as_root=True)

    info = [line.split('|') for line in out.splitlines()]

    if len(info) != 2:
        raise RuntimeError(_("Path %s must be LVM logical volume") % path)

    return dict(zip(*info))


def get_volume_size(path):
    """Get logical volume size in bytes.

    :param path: logical volume path
    :raises: processutils.ProcessExecutionError if getting the volume size
             fails in some unexpected way.
    :raises: exception.VolumeBDMPathNotFound if the volume path does not exist.
    """
    try:
        out, _err = utils.execute('blockdev', '--getsize64', path,
                                  run_as_root=True)
    except processutils.ProcessExecutionError:
        if not utils.path_exists(path):
            raise exception.VolumeBDMPathNotFound(path=path)
        else:
            raise
    return int(out)


def _zero_volume(path, volume_size):
    """Write zeros over the specified path

    :param path: logical volume path
    :param size: number of zeros to write
    """
    bs = units.Mi
    direct_flags = ('oflag=direct',)
    sync_flags = ()
    remaining_bytes = volume_size

    # The loop efficiently writes zeros using dd,
    # and caters for versions of dd that don't have
    # the easier to use iflag=count_bytes option.
    while remaining_bytes:
        zero_blocks = remaining_bytes // bs
        seek_blocks = (volume_size - remaining_bytes) // bs
        zero_cmd = ('dd', 'bs=%s' % bs,
                    'if=/dev/zero', 'of=%s' % path,
                    'seek=%s' % seek_blocks, 'count=%s' % zero_blocks)
        zero_cmd += direct_flags
        zero_cmd += sync_flags
        if zero_blocks:
            utils.execute(*zero_cmd, run_as_root=True)
        remaining_bytes %= bs
        bs //= units.Ki  # Limit to 3 iterations
        # Use O_DIRECT with initial block size and fdatasync otherwise
        direct_flags = ()
        sync_flags = ('conv=fdatasync',)


def clear_volume(path):
    """Obfuscate the logical volume.

    :param path: logical volume path
    """
    # If using thin volumes it doesn't make sense to clear them.
    if CONF.libvirt.thin_logical_volumes:
        return

    volume_clear = CONF.libvirt.volume_clear

    if volume_clear == 'none':
        return

    volume_clear_size = int(CONF.libvirt.volume_clear_size) * units.Mi

    try:
        volume_size = get_volume_size(path)
    except exception.VolumeBDMPathNotFound:
        LOG.warning('ignoring missing logical volume %(path)s', {'path': path})
        return

    if volume_clear_size != 0 and volume_clear_size < volume_size:
        volume_size = volume_clear_size

    if volume_clear == 'zero':
        # NOTE(p-draigbrady): we could use shred to do the zeroing
        # with -n0 -z, however only versions >= 8.22 perform as well as dd
        _zero_volume(path, volume_size)
    elif volume_clear == 'shred':
        utils.execute('shred', '-n3', '-s%d' % volume_size, path,
                      run_as_root=True)


def remove_volumes(paths):
    """Remove one or more logical volume."""

    errors = []
    for path in paths:
        clear_volume(path)
        lvremove = ('lvremove', '-f', path)
        try:
            utils.execute(*lvremove, attempts=3, run_as_root=True)
        except processutils.ProcessExecutionError as exp:
            errors.append(six.text_type(exp))
    if errors:
        raise exception.VolumesNotRemoved(reason=(', ').join(errors))


# WRS: Enable instance resizing for LVM backed instances
def get_volume_vg(path):
    """Get logical volume's volume group name.

    :param path: logical volume path
    """
    lv_info = volume_info(path)
    vg = lv_info['VG']
    return vg


def rename_volume(lv_name, lv_new_name):
    """Rename an LVM image.

    Rename an LVM image.

    :param vg: existing volume group which holds the image volume
    :param lv_name: current name for this image (logical volume)
    :param lv_new_name: bew name for this image (logical volume)
    """
    vg = get_volume_vg(lv_name)
    errors = []
    lvrename = ('lvrename', vg, lv_name, lv_new_name)
    try:
        utils.execute(*lvrename, run_as_root=True, attempts=3)
    except processutils.ProcessExecutionError as exp:
        errors.append(six.text_type(exp))
        if errors:
            raise exception.ResizeError(reason=(', ').join(errors))


def create_duplicate_volume(lv_name, lv_new_name):
    """Duplicate a reference logical volume.

    Creates an LVM volume the same size as a reference volume and with the
    same contents.

    :param lv_name: path of the reference image (logical volume)
    :param lv_new_name: path for the new logical volume
    """
    vg = get_volume_vg(lv_name)
    size = get_volume_size(lv_name)
    errors = []
    try:
        if CONF.libvirt.thin_logical_volumes:
            # Special-case for thin volumes
            name = os.path.basename(lv_name)
            new_name = os.path.basename(lv_new_name)
            thin_copy_volume(name, new_name, vg, size)
        else:
            create_volume(vg, lv_new_name, size, sparse=False)

            # copy the preserved volume contents to the new volume
            utils.copy_image(lv_name, lv_new_name)
    except processutils.ProcessExecutionError as exp:
        errors.append(six.text_type(exp))
        if errors:
            raise exception.ResizeError(reason=(', ').join(errors))


def resize_volume(lv_name, size):
    """Resizes an LVM image.

    Resizes an LVM image to the requested new size.

    :param lv_name: name for the image to be resized (logical volume)
    :param size: new size in bytes for the image (logical volume)
    """
    sizeInMB = size / units.Mi
    errors = []
    lvresize = ('lvresize', '--size', sizeInMB, lv_name)
    try:
        utils.execute(*lvresize, attempts=3, run_as_root=True)
    except processutils.ProcessExecutionError as exp:
        errors.append(six.text_type(exp))
        if errors:
            raise exception.ResizeError(reason=(', ').join(errors))
