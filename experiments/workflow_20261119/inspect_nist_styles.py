from zipfile import ZipFile
import xml.etree.ElementTree as E
from collections import Counter
n={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile('work/thermal_extension_raw/nist_ignition.xlsx') as z:
 print(z.namelist())
 print(z.read('xl/styles.xml').decode()[:9000])
 root=E.fromstring(z.read('xl/worksheets/sheet1.xml'))
 print(Counter(c.get('s') for c in root.findall('.//m:c',n)))
