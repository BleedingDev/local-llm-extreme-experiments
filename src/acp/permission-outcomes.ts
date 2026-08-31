export type AcpToolFailureOutcome = "cancelled" | "permission_rejected" | "failed";

export const acpFailureOutcomeFor = (input: {
  cancelled: boolean;
  message: string;
}): AcpToolFailureOutcome => {
  if (input.cancelled) {
    return "cancelled";
  }
  return input.message.includes("permission rejected") ? "permission_rejected" : "failed";
};

export const acpFailureRawOutput = <T extends Record<string, unknown>>(
  base: T,
  input: {
    cancelled: boolean;
    message: string;
  },
): T & { outcome: AcpToolFailureOutcome } => ({
  ...base,
  outcome: acpFailureOutcomeFor(input),
});
