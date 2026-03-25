import {
  addDays,
  differenceInCalendarDays,
  formatISO,
  isValid,
  parse,
  parseISO,
  startOfDay
} from "date-fns";

const DATE_FORMATS = [
  "MM/dd/yyyy",
  "MM/dd/yy",
  "M/d/yyyy",
  "M/d/yy",
  "dd/MM/yyyy",
  "dd/MM/yy",
  "d/M/yyyy",
  "d/M/yy",
  "yyyy-MM-dd",
  "MMM d yyyy",
  "MMMM d yyyy"
];

export const DATE_REGEXES = [
  /\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b/g,
  /\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b/g,
  /\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b/gi
];

export function normalizeCandidateDate(raw: string): Date | null {
  const trimmed = raw.replace(/best by|use by|sell by|exp|expiry|expires/gi, "").trim();
  for (const format of DATE_FORMATS) {
    const parsed = parse(trimmed, format, new Date());
    if (isValid(parsed)) {
      return startOfDay(parsed);
    }
  }

  const parsed = new Date(trimmed);
  return isValid(parsed) ? startOfDay(parsed) : null;
}

export function extractDates(text: string): Date[] {
  const matches = DATE_REGEXES.flatMap((regex) => text.match(regex) ?? []);
  return matches
    .map((match) => normalizeCandidateDate(match))
    .filter((date): date is Date => Boolean(date));
}

export function inferExpiryDate(referenceDate: Date, shelfLifeDays: number): Date {
  return addDays(startOfDay(referenceDate), shelfLifeDays);
}

export function daysUntil(date: Date, from = new Date()): number {
  return differenceInCalendarDays(startOfDay(date), startOfDay(from));
}

export function formatDate(date: Date | null | undefined): string | null {
  return date ? formatISO(date, { representation: "date" }) : null;
}

export function toDate(value: string | Date | null | undefined): Date | null {
  if (!value) {
    return null;
  }

  if (value instanceof Date) {
    return value;
  }

  const parsed = parseISO(value);
  return isValid(parsed) ? parsed : null;
}
