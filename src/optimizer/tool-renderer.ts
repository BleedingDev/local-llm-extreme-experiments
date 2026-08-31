import { createHash } from "node:crypto";
import type { ResolvedOptimizerPolicy } from "./policy-resolver";
import {
  RenderedToolContractSchema,
  type CanonicalToolSpec,
  type JsonValue,
  type OptimizerRegistryRecord,
  type RenderedToolContract,
} from "./types";

export const DEFAULT_TOOL_RENDERER_ID = "renderer.default";
export const DEFAULT_TOOL_RENDERER_VERSION = "renderer.v1";

type JsonObject = { [key: string]: JsonValue };
type RenderedToolContractRecord = Extract<OptimizerRegistryRecord, { recordKind: "rendered_tool_contract" }>;

export interface RenderToolContractsInput {
  canonicalToolSpecs: readonly CanonicalToolSpec[];
  resolvedPolicy: ResolvedOptimizerPolicy;
}

export interface SelectRenderedToolContractsInput extends RenderToolContractsInput {
  records: readonly OptimizerRegistryRecord[];
}

const stableJsonValue = (value: JsonValue): JsonValue => {
  if (Array.isArray(value)) {
    return value.map((entry) => stableJsonValue(entry));
  }

  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, stableJsonValue(entry)]),
    );
  }

  return value;
};

const stableJson = (value: JsonValue): string => JSON.stringify(stableJsonValue(value));

const shortHash = (value: JsonValue): string =>
  createHash("sha256").update(stableJson(value)).digest("hex").slice(0, 12);

const compareCanonicalSpecs = (left: CanonicalToolSpec, right: CanonicalToolSpec): number =>
  left.namespace.localeCompare(right.namespace) ||
  left.name.localeCompare(right.name) ||
  left.canonicalToolId.localeCompare(right.canonicalToolId);

const renderedToolName = (spec: CanonicalToolSpec): string => `${spec.namespace}_${spec.name}`;

const renderedToolId = (spec: CanonicalToolSpec, resolvedPolicy: ResolvedOptimizerPolicy): string =>
  [
    "rendered",
    spec.canonicalToolId,
    resolvedPolicy.modelProfileId,
    shortHash({
      canonicalToolId: spec.canonicalToolId,
      modelProfileId: resolvedPolicy.modelProfileId,
      policyId: resolvedPolicy.policyId,
      renderer: DEFAULT_TOOL_RENDERER_ID,
      rendererVersion: DEFAULT_TOOL_RENDERER_VERSION,
      renderedToolVersion: resolvedPolicy.renderedToolVersion,
      resultStyleVersion: resolvedPolicy.resultStyleVersion,
      toolCallingMode: resolvedPolicy.modelProfile.toolCallingMode,
    }),
  ].join(".");

const schemaRecord = (value: JsonObject): JsonObject => stableJsonValue(value) as JsonObject;

const stringArray = (value: JsonValue | undefined): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string").sort((left, right) => left.localeCompare(right)) : [];

const schemaTypeSummary = (schema: JsonValue | undefined): string => {
  if (schema === undefined || schema === null || typeof schema !== "object" || Array.isArray(schema)) {
    return "value";
  }

  const type = schema.type;
  if (typeof type === "string") {
    return type;
  }
  if (Array.isArray(type) && type.every((entry) => typeof entry === "string")) {
    return [...type].sort((left, right) => left.localeCompare(right)).join("|");
  }
  if ("properties" in schema) {
    return "object";
  }
  if ("items" in schema) {
    return "array";
  }
  return "value";
};

const descriptionSummary = (schema: JsonValue | undefined): string | undefined => {
  if (schema === undefined || schema === null || typeof schema !== "object" || Array.isArray(schema)) {
    return undefined;
  }
  return typeof schema.description === "string" ? schema.description : undefined;
};

const textFallbackSchema = (schema: JsonObject): JsonObject => {
  const required = stringArray(schema.required);
  const properties = schema.properties;
  const compactProperties =
    properties !== undefined && properties !== null && typeof properties === "object" && !Array.isArray(properties)
      ? Object.fromEntries(
          Object.entries(properties)
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([key, property]) => {
              const compactProperty: JsonObject = { type: schemaTypeSummary(property) };
              const description = descriptionSummary(property);
              if (description !== undefined) {
                compactProperty.description = description;
              }
              return [key, compactProperty];
            }),
        )
      : {};

  const compact: JsonObject = {
    type: "object",
    properties: compactProperties,
  };

  if (required.length > 0) {
    compact.required = required;
  }

  return schemaRecord(compact);
};

const isTextFallback = (resolvedPolicy: ResolvedOptimizerPolicy): boolean =>
  resolvedPolicy.modelProfile.toolCallingMode === "text" ||
  resolvedPolicy.modelProfile.structuredOutputMode === "text" ||
  resolvedPolicy.modelProfile.promptStyle === "plain_text";

const argumentList = (schema: JsonObject): string => {
  const required = new Set(stringArray(schema.required));
  const properties = schema.properties;
  if (properties === undefined || properties === null || typeof properties !== "object" || Array.isArray(properties)) {
    return "no structured arguments";
  }

  const args = Object.entries(properties)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, property]) => `${name}:${schemaTypeSummary(property)}${required.has(name) ? "!" : "?"}`);

  return args.length === 0 ? "no structured arguments" : args.join(", ");
};

const renderDescription = (
  spec: CanonicalToolSpec,
  resolvedPolicy: ResolvedOptimizerPolicy,
  inputSchema: JsonObject,
): string => {
  if (isTextFallback(resolvedPolicy)) {
    return `${spec.title}. ${spec.description} Arguments: ${argumentList(inputSchema)}. Result style: ${spec.resultStyle}.`;
  }
  return `${spec.title}. ${spec.description}`;
};

const renderPromptFragments = (
  spec: CanonicalToolSpec,
  resolvedPolicy: ResolvedOptimizerPolicy,
  inputSchema: JsonObject,
): string[] => {
  if (!isTextFallback(resolvedPolicy)) {
    return [];
  }

  return [
    `Text contract for ${renderedToolName(spec)}: pass one compact JSON object with ${argumentList(inputSchema)}. Do not invent fields.`,
  ];
};

const renderExamples = (
  spec: CanonicalToolSpec,
  resolvedPolicy: ResolvedOptimizerPolicy,
): RenderedToolContract["examples"] => {
  const examples = spec.examples.map((example) => ({
    input: schemaRecord(example.input),
    ...(example.output === undefined ? {} : { expectedResultShape: stableJsonValue(example.output) }),
  }));

  if (isTextFallback(resolvedPolicy)) {
    return examples.slice(0, 1);
  }

  return examples;
};

export const renderToolContract = (
  spec: CanonicalToolSpec,
  resolvedPolicy: ResolvedOptimizerPolicy,
): RenderedToolContract => {
  const inputSchema = isTextFallback(resolvedPolicy)
    ? textFallbackSchema(spec.inputSchema)
    : schemaRecord(spec.inputSchema);

  return RenderedToolContractSchema.parse({
    renderedToolId: renderedToolId(spec, resolvedPolicy),
    canonicalToolId: spec.canonicalToolId,
    canonicalToolVersion: resolvedPolicy.canonicalToolVersion,
    renderedToolVersion: resolvedPolicy.renderedToolVersion,
    modelProfileId: resolvedPolicy.modelProfileId,
    policyId: resolvedPolicy.policyId,
    renderer: DEFAULT_TOOL_RENDERER_ID,
    rendererVersion: DEFAULT_TOOL_RENDERER_VERSION,
    name: renderedToolName(spec),
    description: renderDescription(spec, resolvedPolicy, inputSchema),
    inputSchema,
    resultStyle: spec.resultStyle,
    resultStyleVersion: resolvedPolicy.resultStyleVersion,
    promptFragments: renderPromptFragments(spec, resolvedPolicy, inputSchema),
    examples: renderExamples(spec, resolvedPolicy),
  });
};

export const renderToolContracts = (input: RenderToolContractsInput): RenderedToolContract[] =>
  [...input.canonicalToolSpecs]
    .sort(compareCanonicalSpecs)
    .map((spec) => renderToolContract(spec, input.resolvedPolicy));

const compareRenderedRecordPreference = (
  left: RenderedToolContractRecord,
  right: RenderedToolContractRecord,
): number =>
  right.updatedAt.localeCompare(left.updatedAt) ||
  left.registryRecordId.localeCompare(right.registryRecordId);

const matchesRenderedPolicy = (
  record: RenderedToolContractRecord,
  resolvedPolicy: ResolvedOptimizerPolicy,
  canonicalToolIds: ReadonlySet<string>,
): boolean =>
  record.status === "promoted" &&
  canonicalToolIds.has(record.payload.canonicalToolId) &&
  record.payload.modelProfileId === resolvedPolicy.modelProfileId &&
  record.payload.policyId === resolvedPolicy.policyId &&
  record.payload.canonicalToolVersion === resolvedPolicy.canonicalToolVersion &&
  record.payload.renderedToolVersion === resolvedPolicy.renderedToolVersion &&
  record.payload.resultStyleVersion === resolvedPolicy.resultStyleVersion;

export const selectRenderedToolContracts = (input: SelectRenderedToolContractsInput): RenderedToolContract[] => {
  const freshlyRendered = renderToolContracts(input);
  const canonicalToolIds = new Set(freshlyRendered.map((contract) => contract.canonicalToolId));
  const promotedByCanonicalId = new Map<string, RenderedToolContract>();
  const promotedRecords = input.records
    .filter((record): record is RenderedToolContractRecord => record.recordKind === "rendered_tool_contract")
    .filter((record) => matchesRenderedPolicy(record, input.resolvedPolicy, canonicalToolIds))
    .sort(compareRenderedRecordPreference);

  if (promotedRecords.length === 0) {
    return freshlyRendered;
  }

  for (const record of promotedRecords) {
    if (!promotedByCanonicalId.has(record.payload.canonicalToolId)) {
      promotedByCanonicalId.set(record.payload.canonicalToolId, record.payload);
    }
  }

  if (freshlyRendered.some((contract) => !promotedByCanonicalId.has(contract.canonicalToolId))) {
    return freshlyRendered;
  }

  return freshlyRendered.map((contract) => promotedByCanonicalId.get(contract.canonicalToolId) ?? contract);
};
