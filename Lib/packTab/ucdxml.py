# Copyright 2019 Facebook Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division, absolute_import
from lxml import etree, objectify
import zipfile

__all__ = [
    'load_ucdxml',
    'ucdxml_get_repertoire',
]

def _process_element(elt, ucd, attrs=None):
    if elt.tag.endswith('}group'):
        g = elt.attrib
        for elt in elt.getchildren():
            _process_element(elt, ucd, g)
    elif elt.tag.split('}')[1] in ('char', 'noncharacter', 'reserved', 'surrogate'):
        if attrs is None:
            u = dict(elt.attrib)
        else:
            u = dict(attrs)
            u.update(elt.attrib)

        # TODO Handle '#' in values; should be replaced with the codepoint value.
        # TODO Cast to int for integral values.

        if 'cp' in u:
            cp = int(u['cp'], 16)
            del u['cp']
            ucd[cp] = u
        else:
            first_cp = int(u['first-cp'], 16)
            last_cp = int(u['last-cp'], 16)
            del u['first-cp'], u['last-cp']
            for cp in range(first_cp, last_cp + 1):
                ucd[cp] = u

def load_ucdxml(s):
    if hasattr(s, 'read'):
        s =  s.read()
    else:
        if zipfile.is_zipfile(s):
            with zipfile.ZipFile(s) as z:
                with z.open(z.namelist()[0]) as s:
                    s = s.read()
        else:
            with open(s, 'rb') as s:
                s = s.read()

    return objectify.fromstring(s)

def ucdxml_get_repertoire(ucdxml):
    ucd = [None] * 0x110000
    for elt in ucdxml.repertoire.getchildren():
        _process_element(elt, ucd)
    return ucd


if __name__ == "__main__":
    import sys
    load_ucdxml(sys.argv[1])
