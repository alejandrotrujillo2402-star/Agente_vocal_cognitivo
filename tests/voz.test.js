// Pruebas de static/voz.js: node --test tests/voz.test.js (pytest las corre en test_voz_js.py)
const test = require("node:test");
const assert = require("node:assert/strict");
const Voz = require("../static/voz.js");

const voc = Voz.vocabulario(["IPS", "EPS", "UCI", "REPS", "sedes", "nivel de atención", "Manizales", "Dosquebradas",
  "Caldas", "Meta", "Santa Marta", "Pereira", "Bello"], ["Remedios", "Ibagué", "La Dorada", "Puerto Boyacá"]);
const P = (texto, quien = 0, conf = 0.9) => texto.split(" ").map((w, i) => ({w, conf, start: i, end: i + 0.5, quien}));
const T = palabras => palabras.map(p => p.w).join(" ");
const corrige = texto => T(Voz.procesar(P(texto), voc));

test("vocabulario: nombres mal escritos, siglas y municipios partidos", () => {
  assert.equal(corrige("¿cuántas camas uci hay en manizalez y dos quebradas?"), "¿cuántas camas UCI hay en Manizales y Dosquebradas?");
  assert.equal(corrige("sedes de ips y eps en caldas"), "sedes de IPS y EPS en Caldas");
  assert.equal(corrige("hay 2 quebradas en risaralda"), "hay Dosquebradas en risaralda");   // numerals convirtió "dos"
  assert.equal(corrige("el nivel de atencion en santa marta"), "el nivel de atención en Santa Marta");
  assert.equal(corrige("y en pereyra"), "y en Pereira");
});

test("vocabulario extra: solo correcciones inequívocas", () => {
  assert.equal(corrige("los remedios caseros"), "los remedios caseros");
  assert.equal(corrige("y en Ibague"), "y en Ibagué");
  assert.equal(corrige("en la dorada y puerto boyaca"), "en La Dorada y Puerto Boyacá");
});

test("vocabulario: no toca palabras comunes", () => {
  assert.equal(corrige("cuál es la meta de vacunación"), "cuál es la meta de vacunación");
  assert.equal(corrige("es una casa bella a caldas"), "es una casa bella a Caldas");   // "bella" no es Bello; "a" no se pierde
  assert.equal(corrige("los datos del reporte"), "los datos del reporte");
});

test("números dichos en palabras", () => {
  assert.equal(corrige("hay doscientas once sedes"), "hay 211 sedes");
  assert.equal(corrige("son treinta y dos camas"), "son 32 camas");
  assert.equal(corrige("dos mil trescientos pacientes"), "2300 pacientes");
  assert.equal(corrige("un millón de pesos"), "1000000 de pesos");
  assert.equal(corrige("una IPS y mil gracias"), "una IPS y mil gracias");
  assert.equal(corrige("entre 2 3 sedes"), "entre 2 3 sedes");
  assert.equal(corrige("son doscientas once."), "son 211.");
});

test("guarda el original y la confianza mínima", () => {
  const [p] = Voz.procesar([{w: "manizalez,", conf: 0.4, start: 1, end: 2, quien: 0}], voc);
  assert.equal(p.w, "Manizales,");
  assert.equal(p.original, "manizalez,");
  assert.equal(p.conf, 0.4);
});

test("diarización: segmentos de 1 o 2 palabras se unen al hablante vecino", () => {
  const s1 = Voz.estabilizar([...P("cuántas camas hay en", 0), ...P("Manizales y", 1), ...P("Dosquebradas por favor", 0)]);
  assert.equal(s1.length, 1);
  assert.equal(s1[0].quien, 0);
  assert.equal(s1[0].texto, "cuántas camas hay en Manizales y Dosquebradas por favor");
  const s2 = Voz.estabilizar([...P("hola agente buenas tardes", 0), ...P("sí", 1)]);
  assert.deepEqual(s2.map(s => s.quien), [0]);
  const s3 = Voz.estabilizar([...P("cuántas sedes hay en Caldas", 0), ...P("y cuántas en Risaralda", 1)]);
  assert.deepEqual(s3.map(s => s.quien), [0, 1]);   // un cambio real de hablante sí abre línea
  assert.equal(s3[1].inicio, 0);
});
