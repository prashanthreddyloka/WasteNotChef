import { PrismaClient } from "@prisma/client";

declare global {
  var __wasteNotChefPrisma__: PrismaClient | undefined;
}

export const prisma =
  global.__wasteNotChefPrisma__ ??
  new PrismaClient({
    log: process.env.NODE_ENV === "development" ? ["warn", "error"] : ["error"]
  });

if (process.env.NODE_ENV !== "production") {
  global.__wasteNotChefPrisma__ = prisma;
}
