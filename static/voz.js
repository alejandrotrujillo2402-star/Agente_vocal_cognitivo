/* Corrección de la transcripción en vivo (funciones puras: se prueban con node --test).
   - corregir(): palabras parecidas a un término del vocabulario de la sesión -> su forma correcta
     ("manizalez" -> "Manizales", "uci" -> "UCI", "dos quebradas" / "2 quebradas" -> "Dosquebradas").
   - numerosEnPalabras(): "doscientas once" -> "211" cuando Deepgram no lo convirtió.
   - estabilizar(): un cambio de hablante de 1 o 2 palabras no es un hablante nuevo.
   Cada palabra es {w, conf, start, end, quien}; w conserva la puntuación ("¿Cuántas", "Caldas?"). */
(function (raiz) {
  const sinTildes = s => String(s || "").toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
  const DIGITOS = {"0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve", "10": "diez"};
  const texto = w => sinTildes(w).replace(/[^a-z0-9ñ ]+/g, " ").trim();
  const clave = w => { const t = texto(w); return DIGITOS[t] || t; };   // "2 quebradas" se compara como "dos quebradas"
  // nombres de lugar que también son palabras comunes: en minúscula se dejan ("la meta de vacunación")
  const COMUNES = new Set(["meta", "soledad", "paz", "dorada", "union", "victoria", "esperanza", "florida", "granada",
                           "belen", "cesar", "salud", "palmar", "providencia"]);
  const partes = w => { const m = String(w).match(/^([¿¡"'(«]*)(.*?)([.,;:?!"')»…]*)$/); return {ante: m[1], nucleo: m[2], tras: m[3]}; };

  function distancia(a, b, max) {
    if (Math.abs(a.length - b.length) > max) return max + 1;
    let prev = Array.from({length: b.length + 1}, (_, j) => j);
    for (let i = 1; i <= a.length; i++) {
      const fila = [i];
      let minimo = i;
      for (let j = 1; j <= b.length; j++) {
        fila[j] = Math.min(prev[j] + 1, fila[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        minimo = Math.min(minimo, fila[j]);
      }
      if (minimo > max) return max + 1;
      prev = fila;
    }
    return prev[b.length];
  }

  /* términos -> índice. Siglas y palabras comunes solo se corrigen si coinciden exactas (mayúsculas, tildes);
     los nombres propios también por parecido: hasta 1 letra de diferencia (6-8 letras) o 2 (9 o más).
     "extra" (todos los municipios del país) solo corrige lo inequívoco: varias palabras que juntas forman el nombre
     ("dos quebradas") o una palabra que ya viene con mayúscula ("Ibague" -> "Ibagué"); así "remedios" no cambia. */
  function vocabulario(terminos, extra) {
    const exactos = new Map(), propios = [], extras = new Map();
    for (const forma of terminos || []) {
      const k = clave(forma), compacta = k.replace(/ /g, "");
      if (!k) continue;
      for (const x of [k, compacta]) if (!exactos.has(x)) exactos.set(x, forma);
      if (/^[A-ZÁÉÍÓÚÑ][a-záéíóúñü]/.test(forma) && compacta.length >= 6) propios.push({forma, compacta});
    }
    for (const forma of extra || []) {
      const k = clave(forma);
      for (const x of [k, k.replace(/ /g, "")]) if (k && !exactos.has(x) && !extras.has(x)) extras.set(x, forma);
    }
    return {exactos, propios, extras};
  }
  const umbral = n => (n >= 9 ? 2 : n >= 6 ? 1 : 0);

  function buscar(nucleos, voc) {
    const k = nucleos.map(clave).join(" "), compacta = k.replace(/ /g, "");
    if (nucleos.length === 1 && COMUNES.has(k) && nucleos[0] === nucleos[0].toLowerCase()) return null;
    if (voc.exactos.has(k)) return voc.exactos.get(k);
    const extra = voc.extras || new Map();
    if (nucleos.length > 1) return voc.exactos.get(compacta) || extra.get(k) || extra.get(compacta) || null;   // "dos quebradas" -> "Dosquebradas"
    if (extra.has(k) && /^[A-ZÁÉÍÓÚÑ]/.test(nucleos[0])) return extra.get(k);
    let mejor = null, dMejor = 99;
    for (const p of voc.propios) {
      if (p.compacta[0] !== compacta[0]) continue;
      const d = distancia(compacta, p.compacta, umbral(p.compacta.length));
      if (d <= umbral(p.compacta.length) && d < dMejor) { mejor = p.forma; dMejor = d; }
    }
    return mejor;
  }

  function corregir(palabras, voc) {
    if (!voc || !palabras.length) return palabras;
    const out = [];
    for (let i = 0; i < palabras.length;) {
      let hecho = false;
      for (let n = Math.min(3, palabras.length - i); n >= 1 && !hecho; n--) {
        const grupo = palabras.slice(i, i + n), primera = partes(grupo[0].w), ultima = partes(grupo[n - 1].w);
        const nucleos = grupo.map((p, j) => (j === 0 ? primera : j === n - 1 ? ultima : partes(p.w)).nucleo);
        if (nucleos.some(x => !x)) continue;
        const forma = buscar(nucleos, voc);
        if (!forma) continue;
        const original = grupo.map(p => p.w).join(" "), w = primera.ante + forma + ultima.tras;
        out.push({...grupo[0], w, end: grupo[n - 1].end, conf: Math.min(...grupo.map(p => p.conf ?? 1)),
                  ...(w !== original ? {original} : {})});
        i += n;
        hecho = true;
      }
      if (!hecho) out.push(palabras[i++]);
    }
    return out;
  }

  /* ---- números dichos en palabras ---- */
  const VALOR = {cero: 0, un: 1, uno: 1, una: 1, dos: 2, tres: 3, cuatro: 4, cinco: 5, seis: 6, siete: 7, ocho: 8, nueve: 9,
    diez: 10, once: 11, doce: 12, trece: 13, catorce: 14, quince: 15, dieciseis: 16, diecisiete: 17, dieciocho: 18, diecinueve: 19,
    veinte: 20, veintiun: 21, veintiuno: 21, veintiuna: 21, veintidos: 22, veintitres: 23, veinticuatro: 24, veinticinco: 25,
    veintiseis: 26, veintisiete: 27, veintiocho: 28, veintinueve: 29, treinta: 30, cuarenta: 40, cincuenta: 50, sesenta: 60,
    setenta: 70, ochenta: 80, noventa: 90, cien: 100, ciento: 100, doscientos: 200, doscientas: 200, trescientos: 300,
    trescientas: 300, cuatrocientos: 400, cuatrocientas: 400, quinientos: 500, quinientas: 500, seiscientos: 600, seiscientas: 600,
    setecientos: 700, setecientas: 700, ochocientos: 800, ochocientas: 800, novecientos: 900, novecientas: 900};
  const MULT = {mil: 1e3, millon: 1e6, millones: 1e6};
  const esNumero = k => k in VALOR || k in MULT;

  function valor(claves) {
    let total = 0, actual = 0;
    for (const k of claves) {
      if (k === "y") continue;
      if (k in VALOR) actual += VALOR[k];
      else if (k === "mil") { total += (actual || 1) * 1e3; actual = 0; }
      else { total = ((total + actual) || 1) * 1e6; actual = 0; }
    }
    return total + actual;
  }

  function numerosEnPalabras(palabras) {
    const out = [];
    for (let i = 0; i < palabras.length;) {
      let j = i;
      const claves = [];
      while (j < palabras.length) {
        const k = texto(partes(palabras[j].w).nucleo), sig = j + 1 < palabras.length ? VALOR[texto(partes(palabras[j + 1].w).nucleo)] : 0;
        const previo = VALOR[claves[claves.length - 1]];
        const yValida = k === "y" && previo >= 30 && previo % 10 === 0 && sig >= 1 && sig <= 9;   // "treinta y dos"
        if (!(esNumero(k) || yValida)) break;
        claves.push(k);
        j++;
        if (/[.,;:?!]$/.test(palabras[j - 1].w)) break;   // la puntuación corta la cifra
      }
      const sola = claves.length === 1 && (["un", "una", "uno", "mil"].includes(claves[0]));
      if (j > i && !sola) {
        const w = partes(palabras[i].w).ante + valor(claves) + partes(palabras[j - 1].w).tras;
        out.push({...palabras[i], w, end: palabras[j - 1].end, conf: Math.min(...palabras.slice(i, j).map(p => p.conf ?? 1)),
                  original: palabras.slice(i, j).map(p => p.w).join(" ")});
        i = j;
      } else out.push(palabras[i++]);
    }
    return out;
  }

  /* ---- diarización estable ---- */
  function segmentar(palabras) {
    const segs = [];
    for (const p of palabras) {
      const u = segs[segs.length - 1];
      if (u && u.quien === p.quien) u.palabras.push(p); else segs.push({quien: p.quien, palabras: [p]});
    }
    return segs;
  }

  function estabilizar(palabras) {
    const segs = segmentar(palabras);
    for (let i = 1; i < segs.length - 1; i++)   // A · (1-2 palabras de B) · A  ->  todo A
      if (segs[i].palabras.length < 3 && segs[i - 1].quien === segs[i + 1].quien) segs[i].quien = segs[i - 1].quien;
    for (let i = 0; i < segs.length; i++)       // un cambio de 1-2 palabras no abre línea: se queda con el vecino
      if (segs[i].palabras.length < 3 && segs.length > 1) segs[i].quien = i > 0 ? segs[i - 1].quien : segs[i + 1].quien;
    const unidos = segmentar(segs.flatMap(s => s.palabras.map(p => ({...p, quien: s.quien}))));
    return unidos.map(s => ({quien: s.quien, palabras: s.palabras, texto: s.palabras.map(p => p.w).join(" "),
      inicio: s.palabras[0].start, fin: s.palabras[s.palabras.length - 1].end,
      confianza: s.palabras.reduce((a, p) => a + (p.conf ?? 1), 0) / s.palabras.length}));
  }

  const procesar = (palabras, voc) => numerosEnPalabras(corregir(palabras, voc));
  const Voz = {vocabulario, corregir, numerosEnPalabras, estabilizar, procesar, distancia, sinTildes};
  if (typeof module !== "undefined" && module.exports) module.exports = Voz;
  else raiz.Voz = Voz;
})(typeof window !== "undefined" ? window : globalThis);
