import type { Metadata } from "next";
import { ReactNode } from "react";

import "@copilotkit/react-ui/styles.css";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "MathMate Copilot",
  description: "Interact with the MathMate LangGraph agent through CopilotKit.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
