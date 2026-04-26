
import { Patient, XAIFactor } from "../types";

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";

// Helper for delay/sleep
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Retry utility for rate limiting or transient errors
async function callGroqWithRetry<T>(fn: () => Promise<T>, retries = 4, delay = 2500): Promise<T> {
  try {
    return await fn();
  } catch (error: any) {
    const errorMessage = error?.message || '';
    const errorStatus = error?.status || error?.code;

    const isRateLimit =
      errorMessage.includes('429') ||
      errorMessage.includes('rate_limit') ||
      errorMessage.includes('quota') ||
      errorStatus === 429;

    if (isRateLimit && retries > 0) {
      console.warn(`Groq rate limit hit. Retrying in ${delay}ms... (${retries} retries left)`);
      await sleep(delay + Math.random() * 500);
      return callGroqWithRetry(fn, retries - 1, delay * 2);
    }
    throw error;
  }
}

async function groqChat(model: string, prompt: string, jsonMode = false): Promise<string> {
  const apiKey = process.env.GROQ_API_KEY || process.env.API_KEY || '';
  const res = await fetch(GROQ_API_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      ...(jsonMode && { response_format: { type: "json_object" } }),
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Groq API ${res.status}: ${err}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "";
}

export const runMLInference = async (rawData: any): Promise<Patient> => {
  const prompt = `
    You are a Machine Learning model specialized in the MIMIC-III dataset.
    Input Raw Data: ${JSON.stringify(rawData)}

    Perform the following tasks:
    1. Calculate Sepsis Risk (0-1).
    2. Estimate LOS in days.
    3. Calculate SOFA, GCS, and APACHE II.
    4. Provide clinical diagnosis.
    5. Identify Pre-existing conditions and Allergies.
    6. Generate Medication Chart (name, dosage, frequency, route, status, durationDays).

    Return JSON schema:
    {
      "id": "string",
      "name": "string",
      "age": number,
      "gender": "string",
      "admissionTime": "string",
      "vitals": { "heartRate": number, "systolicBP": number, "temperature": number, "respiratoryRate": number, "oxygenSaturation": number },
      "labs": { "wbc": number, "lactate": number, "creatinine": number, "crp": number, "platelets": number, "bilirubin": number },
      "clinicalScores": { "sofa": number, "gcs": number, "apacheII": number },
      "allergies": ["string"],
      "preExistingConditions": ["string"],
      "medications": [{ "name": "string", "dosage": "string", "frequency": "string", "route": "string", "status": "Active", "durationDays": number }],
      "sepsisRisk": number,
      "predictedLOS": number,
      "diagnosis": "string",
      "acuityLevel": "Critical" | "High" | "Moderate" | "Low"
    }
  `;

  return callGroqWithRetry(async () => {
    const text = await groqChat("llama-3.3-70b-versatile", prompt, true);
    return JSON.parse(text);
  });
};

export const getClinicalSummary = async (patient: Patient): Promise<{ highlights: string[]; fullSummary: string }> => {
  const prompt = `
    Summarize ICU case for ${patient.name}.
    DIAGNOSIS: ${patient.diagnosis}
    RISK: ${(patient.sepsisRisk * 100).toFixed(0)}% Sepsis Risk.

    Return JSON: { "highlights": ["Point 1", "Point 2"], "fullSummary": "..." }
  `;

  try {
    return await callGroqWithRetry(async () => {
      const text = await groqChat("llama-3.1-8b-instant", prompt, true);
      const result = JSON.parse(text);
      return {
        highlights: result.highlights || ["Patient requires close monitoring."],
        fullSummary: result.fullSummary || "No summary available."
      };
    });
  } catch (error) {
    console.error("Clinical Summary Error after retries:", error);
    return {
      highlights: [
        "Rate Limit: Using Heuristic Summary",
        `High Sepsis Risk identified (${(patient.sepsisRisk * 100).toFixed(0)}%).`,
        `Predicted ICU Stay: ${patient.predictedLOS} days.`
      ],
      fullSummary: "The AI summary engine is currently experiencing high demand. Clinical review of raw data is advised. Patient shows signs consistent with " + patient.diagnosis + "."
    };
  }
};

function getHeuristicXAI(patient: Patient): XAIFactor[] {
  return [
    { feature: "WBC Count (Heuristic)", contribution: patient.labs.wbc > 12 ? 0.75 : 0.1, impact: "High" },
    { feature: "Lactate Elevation", contribution: patient.labs.lactate > 2 ? 0.9 : 0.05, impact: "High" },
    { feature: "Oxygen Saturation", contribution: patient.vitals.oxygenSaturation < 92 ? 0.6 : -0.2, impact: "Medium" },
    { feature: "Age Factor", contribution: patient.age > 70 ? 0.4 : 0.1, impact: "Low" },
    { feature: "Clinical Score (SOFA)", contribution: patient.clinicalScores.sofa / 15, impact: "Medium" }
  ];
}

function parseXAIResponse(parsed: unknown, patient: Patient): XAIFactor[] {
  let items: XAIFactor[] = [];
  if (Array.isArray(parsed) && parsed.length > 0) {
    items = parsed.filter((item): item is XAIFactor =>
      item && typeof item === "object" && "feature" in item && "contribution" in item
    ).map((item) => ({
      feature: String(item.feature ?? ""),
      contribution: Number(item.contribution) || 0,
      impact: ["High", "Medium", "Low"].includes(String(item.impact)) ? item.impact as "High" | "Medium" | "Low" : "Medium"
    }));
  } else if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>;
    const arr = obj.factors ?? obj.data ?? obj.items ?? obj.feedback ?? obj.results;
    if (Array.isArray(arr) && arr.length > 0) {
      items = parseXAIResponse(arr, patient);
    }
  }
  return items.length > 0 ? items : getHeuristicXAI(patient);
}

export const getXAIExplanation = async (patient: Patient): Promise<XAIFactor[]> => {
  const prompt = `
    Explain the factors influencing the ${(patient.sepsisRisk * 100).toFixed(0)}% sepsis risk for ${patient.name}.
    Return JSON with key "factors" containing an array: {"factors": [{"feature": "Simplified Name", "contribution": number (-1 to 1), "impact": "High" | "Medium" | "Low"}]}
  `;

  try {
    const result = await callGroqWithRetry(async () => {
      let text = await groqChat("llama-3.1-8b-instant", prompt, true);
      const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) ?? [null, text];
      text = (jsonMatch[1] ?? text).trim();
      return JSON.parse(text);
    });
    return parseXAIResponse(result, patient);
  } catch (error) {
    console.error("XAI Groq Error after retries:", error);
    return getHeuristicXAI(patient);
  }
};
