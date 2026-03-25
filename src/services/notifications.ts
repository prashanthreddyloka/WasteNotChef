import cron from "node-cron";
import nodemailer from "nodemailer";
import webpush from "web-push";
import { prisma } from "../db/prismaClient";
import { daysUntil, formatDate, toDate } from "../utils/dates";

function getTransport() {
  const { SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS } = process.env;
  if (!SMTP_HOST || !SMTP_PORT || !SMTP_USER || !SMTP_PASS) {
    return null;
  }

  return nodemailer.createTransport({
    host: SMTP_HOST,
    port: Number(SMTP_PORT),
    secure: Number(SMTP_PORT) === 465,
    auth: { user: SMTP_USER, pass: SMTP_PASS }
  });
}

export async function sendReminder(message: string, email?: string, endpoint?: string) {
  const transport = getTransport();
  if (transport && email) {
    await transport.sendMail({
      from: process.env.NOTIFY_EMAIL_FROM,
      to: email,
      subject: "WasteNotChef reminder",
      text: message
    });
  } else {
    console.log(`[email-stub] ${email ?? "no-email"} :: ${message}`);
  }

  const { WEB_PUSH_PUBLIC_KEY, WEB_PUSH_PRIVATE_KEY, WEB_PUSH_SUBJECT } = process.env;
  if (WEB_PUSH_PUBLIC_KEY && WEB_PUSH_PRIVATE_KEY && endpoint) {
    webpush.setVapidDetails(WEB_PUSH_SUBJECT ?? "mailto:hello@wastenotchef.local", WEB_PUSH_PUBLIC_KEY, WEB_PUSH_PRIVATE_KEY);
    await webpush.sendNotification(JSON.parse(endpoint), JSON.stringify({ title: "WasteNotChef", body: message }));
  } else {
    console.log(`[push-stub] ${message}`);
  }
}

export function startReminderScheduler() {
  cron.schedule("0 */6 * * *", async () => {
    const prefs = await prisma.reminderPreference.findFirst({ where: { enabled: true } });
    const threshold = prefs?.reminderDays ?? Number(process.env.REMINDER_DAYS ?? 2);
    const items = await prisma.pantryItem.findMany({ where: { status: "active" } });
    const expiringSoon = items.filter((item) => {
      const expiry = toDate(item.detectedExpiry ?? item.inferredExpiry);
      return expiry ? daysUntil(expiry) <= threshold : false;
    });

    for (const item of expiringSoon) {
      const expiry = toDate(item.detectedExpiry ?? item.inferredExpiry);
      await sendReminder(
        `${item.name} is expiring on ${formatDate(expiry)}. Add it to your quest plan soon.`,
        prefs?.email ?? undefined,
        prefs?.webPushEndpoint ?? undefined
      );
    }
  });
}
