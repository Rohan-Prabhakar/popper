import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { CampaignProvider } from "@/lib/campaign-context";
import { AppShell } from "@/components/app-shell";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Popper RL Validation — Dashboard",
  description:
    "Adaptive validation dashboard for open-weight LLMs (Mistral, Qwen, Llama, and more). Extension of the Popper framework.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased`}>
        <CampaignProvider>
          <AppShell>{children}</AppShell>
        </CampaignProvider>
      </body>
    </html>
  );
}
