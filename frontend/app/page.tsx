"use client";

import { CopilotChat } from "@copilotkit/react-ui";
import { type CSSProperties, useMemo } from "react";

const heroStyles: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "1rem",
  maxWidth: "40rem",
};

export default function HomePage() {
  const instructions = useMemo(
    () =>
      "You are MathMate, a helpful LangGraph-powered tutor. " +
      "Leverage OCR, retrieval, and math reasoning to craft explanations.",
    [],
  );

  return (
    <main
      style={{
        display: "flex",
        minHeight: "100vh",
        alignItems: "stretch",
        justifyContent: "center",
        padding: "3rem 1.5rem",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "1200px",
          display: "grid",
          gap: "2rem",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
        }}
      >
        <section style={heroStyles}>
          <p
            style={{
              textTransform: "uppercase",
              fontSize: "0.875rem",
              letterSpacing: "0.2em",
              color: "#3b82f6",
            }}
          >
            MathMate Copilot
          </p>
          <h1 style={{ fontSize: "2.5rem", margin: 0 }}>
            Collaborative math tutoring with LangGraph
          </h1>
          <p style={{ fontSize: "1.125rem", lineHeight: 1.6 }}>
            Chat with the multi-agent MathMate workflow directly from your browser.
            CopilotKit streams intermediate results so you can follow along as the
            agents retrieve context, run OCR, and craft step-by-step explanations.
          </p>
          <ul style={{ margin: 0, paddingLeft: "1.25rem", lineHeight: 1.6 }}>
            <li>Upload problem statements or images via the chat composer.</li>
            <li>Request alternative solution strategies or generate practice sets.</li>
            <li>Inspect the retrieved textbook and video references instantly.</li>
          </ul>
        </section>
        <section
          style={{
            backgroundColor: "rgba(255,255,255,0.85)",
            borderRadius: "24px",
            boxShadow: "0 30px 60px rgba(15, 23, 42, 0.15)",
            backdropFilter: "blur(12px)",
            padding: "1rem",
            minHeight: "540px",
            display: "flex",
          }}
        >
          <CopilotChat
            instructions={instructions}
            placeholder="Ask MathMate anything about the problem you're studying..."
            labels={{
              title: "MathMate",
              description:
                "An orchestrated tutor combining OCR, RAG, and math reasoning",
            }}
            className="mathmate-chat"
          />
        </section>
      </div>
    </main>
  );
}
