# WiFi channel state information parser 

Implementation in Python for the parsing of raw CSI data files. 
- Full support for static file parsing with Intel IWL5300 
- Partial support for Atheros Ath9k devices and stream parsing with IWL5300 

Tool has to be used along with the output produced by one of the modified WiFi kernel drivers provided here:

https://dhalperi.github.io/linux-80211n-csitool/

http://pdcc.ntu.edu.sg/wands/Atheros/

Example: 
```python
        device = WiFiDeviceFactory.create_device("IWL5300")
        csiObj = CSIExtractor(device)
        p = csiObj.open_csi_file('../../data/csi_data_measurement_1.dat')
        csiObj.convert_to_csi_matrix()  
```
**note: multi-dimensional matrix container classes are part of the dataflow framework. 
