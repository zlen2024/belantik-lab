# Pembetulan metrik: kandidat yang sama mesti seri

17 November 2026. Selepas pilot text8, GPT-6.1 Sol Ultra mendapati sesetengah
noise candidate mempunyai word ID yang sama seperti positive context tetapi
skor JAX berbeza sekitar pembundaran apabila positive dan negatives dikira
dalam bentuk batch berbeza. Secara matematik skor untuk pasangan yang sama
adalah sama; raw `==` boleh tersalah kira win/loss dan ranking.

Pembetulan **posthoc pengukuran sahaja**: muatkan 48 checkpoint asal (initial
dan epoch1/2/3 bagi 12 run); guna data validation dan scorer asal. Di setiap
baris, jika `negative_word_id == positive_word_id`, paksa kedua-dua nilai
numeriknya sama sebelum mengira pairwise AUC dengan half-credit dan expected
reciprocal rank bagi ties. Jangan paksa skor token berlainan menjadi sama dan
jangan tambah numeric tolerance. Ini tidak mengubah training, loss, seed,
data, checkpoint, epoch3 endpoint atau reserved test. Retain metrik asal dalam
run asal; rekod metrik dibetulkan, counts/maximum error identity dan hashes
sebagai run audit berasingan. Skor frequency menggunakan satu lookup per word
dan sudah memberi tie tepat untuk ID sama.

Ini membetulkan perbandingan yang memang sama, bukan mengesahkan bahawa dua
perkataan berlainan yang hampir setara secara semantik mestilah tie. Jika
numerical difference antara duplicate ID melebihi 1e-12 atau checkpoint/data
hash gagal, hentikan audit untuk siasatan. Zero-coupling fixture berasingan
masih perlu menguji tie matematik merentas ID yang berlainan.
