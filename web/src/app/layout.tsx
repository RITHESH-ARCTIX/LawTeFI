import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LawTeFI — Legal research, grounded in sources",
  description: "A source-grounded legal research and learning workspace for India.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
