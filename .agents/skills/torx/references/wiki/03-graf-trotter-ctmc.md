# 03. Graf, Trotter dan CTMC

Sumber: notebook 02, 03, 05.

## Jalan rawak pada graf (02)

- Satu nod = satu pbit. Satu tepi = satu `PSWAP`.
- Generator `Q = −L`, `L = Σ_edges w_ij (e_i − e_j)(e_i − e_j)^T`, taburan `p(t) = e^{tQ} p0`.
- Satu lapisan Trotter = hasil darab satu PSWAP setiap tepi. Tepi berkongsi nod tidak komut, jadi hasil darab bertertib berbeza daripada `e^{-Lt}` kecuali had `N → ∞`.
- Ralat dipisah: **splitting bias** (deterministik, daripada urutan) dan **hingar Monte Carlo** (daripada sampel). Strang splitting (urutan maju-undur simetri) menyukat kadar penumpuan.
- Graf terarah: `PJUMP` ganti `PSWAP`.

## Bunny (03)

Jejaring segi tiga Stanford bunny jadi graf. Tiga angka ralat: ralat sampel, splitting bias, jumlah. Rujukan NumPy: eigensolusi tepat dan min sweep bertertib tanpa sampel. "Diffusion" di sini hanya konektiviti graf, bukan haba fizikal.

## Reaksi kimia (05)

- CTMC atas kiraan molekul. Enumerasi konfigurasi boleh capai, satu konfigurasi = satu keadaan, **satu token** (satu bit hidup) bergerak.
- Satu tepi = satu `PJUMP`, kadar daripada propensity mass-action. `pjump_prob(r, dt) = 1 − exp(−r dt)` lalu ke logit.
- Dua rujukan: `e^{Qt}` tepat dan simulasi Gillespie.
- Kos: bilangan konfigurasi membesar dengan cepat, jadi pendekatan ini tidak berskala untuk rangkaian besar.

## Kenapa berkait dengan kita

Corak "satu gate setempat per unit, ulang lapisan" sama seperti cara kita susun satu PNOT per dimensi. Perbezaan: kita tak perlukan Trotter kerana p-bit kita bebas.
