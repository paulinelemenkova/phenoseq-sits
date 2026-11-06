var CRS = 'EPSG:32637';
var TRANSFORM = [30, 0, 477660, 0, -30, 4111590];
var REGION = ee.Geometry.Rectangle([477660, 4059360, 531300, 4111590], CRS, false);
var FOLDER = 'aoi_sits';

function exportGrid(img, name) {
  Export.image.toDrive({
    image: img, description: name, fileNamePrefix: name, folder: FOLDER,
    crs: CRS, crsTransform: TRANSFORM, region: REGION, maxPixels: 1e10
  });
}

var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').toUint8();
exportGrid(worldcover, 'aoi_worldcover_2021');

var wc = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100');
function product(name, season) {
  return wc.filter(ee.Filter.eq('product', name))
           .filter(ee.Filter.eq('season', season))
           .select('classification').max().unmask(0).rename(name + '_' + season);
}
var cereal = product('wintercereals', 'tc-wintercereals')
  .addBands(product('maize', 'tc-maize-main'))
  .addBands(product('maize', 'tc-maize-second'))
  .rename(['wintercereals', 'maize_main', 'maize_second']).toUint8();
exportGrid(cereal, 'aoi_worldcereal_2021');

print('WorldCereal products found:', wc.aggregate_array('product').distinct(),
      wc.aggregate_array('season').distinct());
Map.centerObject(REGION, 10);
Map.addLayer(worldcover.eq(40).selfMask().clip(REGION), {palette: ['f096ff']}, 'WorldCover cropland');
Map.addLayer(cereal.select('wintercereals').selfMask().clip(REGION), {palette: ['e69f00']}, 'Winter cereals 2021');
Map.addLayer(cereal.select('maize_second').selfMask().clip(REGION), {palette: ['009e73']}, 'Second-season maize 2021');
