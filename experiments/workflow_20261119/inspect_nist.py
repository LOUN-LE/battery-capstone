from zipfile import ZipFile
import xml.etree.ElementTree as E
p='work/thermal_extension_raw/nist_ignition.xlsx'
n={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile(p) as z:
 s=[]
 if 'xl/sharedStrings.xml' in z.namelist():
  s=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',n)]
 print(z.read('xl/workbook.xml').decode()[:9000])
 for f in z.namelist():
  if f.startswith('xl/worksheets/sheet') and f.endswith('.xml'):
   root=E.fromstring(z.read(f)); rows=root.findall('m:sheetData/m:row',n)
   print('\nSHEET',f,'ROWS',len(rows))
   for r in rows[:15]:
    vals=[]
    for c in r:
     v=c.find('m:v',n); value=v.text if v is not None else ''.join(c.find('m:is',n).itertext()) if c.find('m:is',n) is not None else ''
     if c.get('t')=='s': value=s[int(value)]
     if value: vals.append((c.get('r'),value))
    print(vals[:35])
 print('\nSHAREDSTRINGS',s[:200])
