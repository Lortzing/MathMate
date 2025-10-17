"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { ReactNode, useMemo } from "react";

const DEFAULT_RUNTIME_URL = "http://localhost:4000/copilotkit";
const DEFAULT_AGENT_NAME = "mathmate";

export function Providers({ children }: { children: ReactNode }) {
  const runtimeUrl =
    process.env.NEXT_PUBLIC_COPILOT_RUNTIME_URL ?? DEFAULT_RUNTIME_URL;
  const agent = process.env.NEXT_PUBLIC_COPILOT_AGENT ?? DEFAULT_AGENT_NAME;
  const publicApiKey = process.env.NEXT_PUBLIC_COPILOT_PUBLIC_API_KEY;

  const providerProps = useMemo(
    () => ({ runtimeUrl, agent, publicApiKey }),
    [agent, publicApiKey, runtimeUrl],
  );

  return (
    <CopilotKit
      runtimeUrl={providerProps.runtimeUrl}
      agent={providerProps.agent}
      publicApiKey={providerProps.publicApiKey || undefined}
    >
      {children}
    </CopilotKit>
  );
}
