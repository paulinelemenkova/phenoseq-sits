var YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var SCALE = 30;
var CRS = 'EPSG:32637';
var FOLDER = 'aoi_sits';
var CS_THRESHOLD = 0.60;
var S1_PASS = 'DESCENDING';
var EXPORT_S2 = true;
var EXPORT_S1 = true;
var EXPORT_L89 = false;
var NODATA = -32768;

var AOI = ee.Geometry.Rectangle([38.75, 36.68, 39.35, 37.15], null, false);

var MONTHS = ee.List.sequence(1, 12);
function pad2(m) { return (m < 10 ? '0' : '') + m; }
function monthNames(prefix) {
  var out = [];
  for (var m = 1; m <= 12; m++) { out.push(prefix + '_' + pad2(m)); }
  return out;
}

function emptyImage(bandNames) {
  var zeros = bandNames.map(function () { return 0; });
  return ee.Image.constant(zeros).rename(bandNames).toFloat().updateMask(0);
}

function yearlyStack(year, bandNames, f) {
  var images = MONTHS.map(function (m) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');
    return f(start, end);
  });
  return ee.ImageCollection.fromImages(images).toBands();
}

function regroup(stack, vars) {
  var src = [], dst = [];
  vars.forEach(function (v) {
    for (var m = 1; m <= 12; m++) {
      src.push((m - 1) + '_' + v);
      dst.push(v + '_' + pad2(m));
    }
  });
  return stack.select(src, dst);
}

function toInt16(img, factor) {
  return img.multiply(factor).round().clamp(-32767, 32767)
            .unmask(NODATA).toInt16().clip(AOI);
}

function exportImage(img, name) {
  Export.image.toDrive({
    image: img,
    description: name,
    folder: FOLDER,
    fileNamePrefix: name,
    region: AOI,
    scale: SCALE,
    crs: CRS,
    maxPixels: 1e10,
    formatOptions: {cloudOptimized: true, noData: NODATA}
  });
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(AOI);
var csPlus = ee.ImageCollection('GOOGLE_CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

function s2Indices(img) {
  var clear = img.select('cs_cdf').gte(CS_THRESHOLD);
  var r = img.select(['B2', 'B4', 'B8']).divide(10000);
  var ndvi = r.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var evi = r.expression('2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)',
      {N: r.select('B8'), R: r.select('B4'), B: r.select('B2')})
      .clamp(-1, 1).rename('EVI');
  return ndvi.addBands(evi).updateMask(clear)
             .copyProperties(img, ['system:time_start']);
}

function s2Month(start, end) {
  var col = s2.filterDate(start, end)
              .linkCollection(csPlus, ['cs_cdf'])
              .map(s2Indices);
  return ee.Image(ee.Algorithms.If(col.size().gt(0),
      col.median(), emptyImage(['NDVI', 'EVI'])));
}

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(AOI)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
if (S1_PASS !== 'BOTH') {
  s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', S1_PASS));
}

function s1Linear(img) {
  var db = img.select(['VV', 'VH']);
  var lin = ee.Image(10).pow(db.divide(10)).updateMask(db.gt(-30));
  return lin.copyProperties(img, ['system:time_start']);
}

function s1Month(start, end) {
  var col = s1.filterDate(start, end).map(s1Linear);
  return ee.Image(ee.Algorithms.If(col.size().gt(0),
      col.median().log10().multiply(10).rename(['VV', 'VH']),
      emptyImage(['VV', 'VH'])));
}

var l89 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    .filterBounds(AOI);

function lNdvi(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0)
      .and(qa.bitwiseAnd(1 << 3).eq(0))
      .and(qa.bitwiseAnd(1 << 4).eq(0));
  var sr = img.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
  return sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
           .updateMask(clear).copyProperties(img, ['system:time_start']);
}

function lMonth(start, end) {
  var col = l89.filterDate(start, end).map(lNdvi);
  return ee.Image(ee.Algorithms.If(col.size().gt(0),
      col.median(), emptyImage(['NDVI'])));
}

function countTable(year) {
  return ee.FeatureCollection(MONTHS.map(function (m) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');
    return ee.Feature(null, {
      year: year, month: m,
      S2: s2.filterDate(start, end).size(),
      S1: s1.filterDate(start, end).size(),
      L89: l89.filterDate(start, end).size()
    });
  }));
}

var tasks = 0;
YEARS.forEach(function (y) {
  print('Acquisitions per month, ' + y, countTable(y));
  if (EXPORT_S2) {
    exportImage(toInt16(regroup(yearlyStack(y, null, s2Month), ['NDVI', 'EVI']), 10000),
                'aoi_s2_ndvi_evi_' + y);
    tasks++;
  }
  if (EXPORT_S1) {
    exportImage(toInt16(regroup(yearlyStack(y, null, s1Month), ['VV', 'VH']), 100),
                'aoi_s1_vv_vh_' + y);
    tasks++;
  }
  if (EXPORT_L89) {
    exportImage(toInt16(regroup(yearlyStack(y, null, lMonth), ['NDVI']), 10000),
                'aoi_l89_ndvi_' + y);
    tasks++;
  }
});

var widthM = 0.60 * 111320 * Math.cos(36.915 * Math.PI / 180);
var heightM = 0.47 * 110570;
var pixels = Math.round(widthM / SCALE) * Math.round(heightM / SCALE);
var bandsPerYear = (EXPORT_S2 ? 24 : 0) + (EXPORT_S1 ? 24 : 0) + (EXPORT_L89 ? 12 : 0);
print('Export tasks queued: ' + tasks);
print('Pixels per band: ' + pixels + ' (' + SCALE + ' m)');
print('Approx. uncompressed size, all years (MB): ' +
      Math.round(pixels * 2 * bandsPerYear * YEARS.length / 1e6));

Map.centerObject(AOI, 10);
var ndviVis = {min: 0, max: 0.8, palette: ['#8c510a', '#f6e8c3', '#c7eae5', '#01665e']};
Map.addLayer(s2Month(ee.Date('2025-07-01'), ee.Date('2025-08-01')).select('NDVI').clip(AOI),
             ndviVis, 'NDVI July 2025');
Map.addLayer(s2Month(ee.Date('2025-09-01'), ee.Date('2025-10-01')).select('NDVI').clip(AOI),
             ndviVis, 'NDVI September 2025');
Map.addLayer(ee.Image().paint(AOI, 0, 2), {palette: ['red']}, 'Analysis window');
