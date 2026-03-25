curl -X POST http://localhost:4000/api/upload-photo ^
  -F "image=@examples/demo-fridge.jpg"

curl -X POST http://localhost:4000/api/recipes/from-items ^
  -H "Content-Type: application/json" ^
  -d "{\"items\":[{\"name\":\"milk\"},{\"name\":\"eggs\"},{\"name\":\"spinach\"}]}"

curl -X POST http://localhost:4000/api/plan/week ^
  -H "Content-Type: application/json" ^
  -d "{\"items\":[{\"id\":\"1\",\"name\":\"milk\",\"detectedExpiry\":\"2026-03-28\"}],\"preferences\":{\"mealsPerDay\":1,\"skipDays\":[],\"preferCuisineTags\":[\"quick\"],\"maxLeftovers\":2}}"
