const PALETTE = [
  "#68BDF6",
  "#6DCE9E",
  "#FF756E",
  "#DE9BF9",
  "#FB95AF",
  "#FFC766",
  "#8DCC93",
  "#4C8EDA",
  "#F16667",
  "#FFB174",
  "#A4DD00",
  "#D9C8AE",
];

export function colorForLabel(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  return PALETTE[Math.abs(hash) % PALETTE.length];
}

export function primaryLabel(labels: string[] | undefined): string {
  if (!labels?.length) return "Entity";
  return labels[0];
}
