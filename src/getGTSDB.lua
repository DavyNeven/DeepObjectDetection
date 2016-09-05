local pl = (require 'pl.import_into')()

local function getDataset(csv_path, nImages)

  csv_path = csv_path or '/esat/larimar/dneven/datasets/GTSDB/FullIJCNN2013/gt.csv'
  nImages = nImages or 900
  local csv = pl.data.read('/esat/larimar/dneven/datasets/GTSDB/FullIJCNN2013/gt.csv')

  -- convert CSV
  local dataset = {}
  for i = 1, nImages do
    local bbox = {}
    table.insert(dataset, i, {string.format("%05d", i - 1) .. '.ppm', bbox})
  end
  --print(dataset)
  for i = 1, #csv do
    local n = tonumber(string.sub(csv[i][1], 1, 5)) + 1 -- one based
    local bbox = {csv[i][2],csv[i][3],csv[i][4],csv[i][5]}
    --print(dataset[n][2])
    table.insert(dataset[n][2], bbox)
  end

  return dataset

end

return getDataset

