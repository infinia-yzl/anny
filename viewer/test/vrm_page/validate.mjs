// OpenSculptBoy
// Apache License, Version 2.0
//
// The Khronos glTF validator on GLB and VRM files (run by test/test_vrm_three.py):
//   node viewer/test/vrm_page/validate.mjs <file> [<file> ...]
// It prints a JSON list with one report per file: the counts of errors, warnings, infos and hints,
// and every message (code, severity, pointer and text). The VRM extensions are unknown to the
// validator, which reports them as infos; it still checks the glTF core of the file in full.
import fs from 'fs';
import path from 'path';
import validator from 'gltf-validator';

const reports = [];
for (const file of process.argv.slice(2)) {
  const report = await validator.validateBytes(new Uint8Array(fs.readFileSync(file)), {
    uri: path.basename(file),
    maxIssues: 0, // every message
  });
  reports.push({
    file,
    validatorVersion: report.validatorVersion,
    numErrors: report.issues.numErrors,
    numWarnings: report.issues.numWarnings,
    numInfos: report.issues.numInfos,
    numHints: report.issues.numHints,
    messages: report.issues.messages,
    extensionsUsed: report.info ? report.info.extensionsUsed : undefined,
  });
}
process.stdout.write(JSON.stringify(reports));
