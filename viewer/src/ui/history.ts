// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Undo and redo of the look (the body, the face, the colours and the hair). An entry holds the look before an edit;
// a slider drag makes one entry, opened by its first input and closed by its change event.

export interface LookHistory {
  begin(label: string): void;
  end(): void;
  undo(): string | null;
  redo(): string | null;
  canUndo(): boolean;
  canRedo(): boolean;
  nextUndo(): string;
  nextRedo(): string;
  onChange: () => void;
}

export function lookHistory(get: () => any, apply: (look: any) => void, limit = 100): LookHistory {
  const undo: { label: string; look: string }[] = [], redo: { label: string; look: string }[] = [];
  let open: string | null = null;
  const H: LookHistory = {
    onChange: () => {},
    begin(label) {
      if (open === label) return;
      open = label;
      const look = JSON.stringify(get());
      if (undo.length && undo[undo.length - 1].look === look && undo[undo.length - 1].label === label) return;
      undo.push({ label, look });
      if (undo.length > limit) undo.shift();
      redo.length = 0;
      H.onChange();
    },
    end() { open = null; },
    undo() {
      open = null;
      // entries of edits that changed nothing are skipped
      const now = JSON.stringify(get());
      while (undo.length && undo[undo.length - 1].look === now) undo.pop();
      const e = undo.pop(); if (!e) { H.onChange(); return null; }
      redo.push({ label: e.label, look: now });
      apply(JSON.parse(e.look));
      H.onChange();
      return e.label;
    },
    redo() {
      open = null;
      const e = redo.pop(); if (!e) return null;
      undo.push({ label: e.label, look: JSON.stringify(get()) });
      apply(JSON.parse(e.look));
      H.onChange();
      return e.label;
    },
    canUndo: () => undo.length > 0,
    canRedo: () => redo.length > 0,
    nextUndo: () => undo.length ? undo[undo.length - 1].label : '',
    nextRedo: () => redo.length ? redo[redo.length - 1].label : '',
  };
  return H;
}
