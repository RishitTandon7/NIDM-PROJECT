import fiona

shp_path = r"F:\NIDM-SRM Drone Project Docs\GLOF project\image analysis\classified_glacier_dt.shp"

try:
    with fiona.open(shp_path) as src:
        print("Schema:", src.schema)
        print("CRS:", src.crs)
        print("Count:", len(src))
        
        # Check first few records
        for i, item in enumerate(src):
            if i >= 5: break
            print(f"Record {i}:", item['properties'])
            
        # Collect unique values for potential class columns
        # Assuming schema properties are the columns
        columns = src.schema['properties'].keys()
        unique_values = {col: set() for col in columns}
        
        for item in src:
            for col in columns:
                val = item['properties'][col]
                if isinstance(val, (str, int)):
                    unique_values[col].add(val)
                    
        print("\nUnique values per column:")
        for col, values in unique_values.items():
            if len(values) < 20: # Only print if few unique values
                print(f"{col}: {values}")
            else:
                print(f"{col}: {len(values)} unique values (too many to list)")
                
except Exception as e:
    print(f"Error reading shapefile: {e}")





