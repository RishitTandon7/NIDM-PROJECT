import geopandas as gpd

shp_path = r"F:\NIDM-SRM Drone Project Docs\GLOF project\image analysis\classified_glacier_dt.shp"

try:
    gdf = gpd.read_file(shp_path)
    print("Columns:", gdf.columns)
    print("First 5 rows:")
    print(gdf.head())
    
    # Try to find a column that might be the class
    for col in gdf.columns:
        if gdf[col].dtype == 'object' or gdf[col].dtype == 'int64':
            print(f"\nUnique values in {col}:")
            print(gdf[col].unique())
except Exception as e:
    print(f"Error reading shapefile: {e}")





