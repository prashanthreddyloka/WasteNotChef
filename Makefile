install:
	npm install
	npm install --prefix client

dev:
	npm run dev

test:
	npm test

seed:
	npm run prisma:seed
