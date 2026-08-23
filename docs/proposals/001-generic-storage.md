# Proposition 001 : Stockage générique (stack / heap / arena) pour tout `ferrite`

**Statut :** proposition ; Phases 0, 1, 2 et 5 appliquées ; Phases 3 et 4 non implémentées
**Date :** 2026-07-31, Phase 2 complétée le 2026-08-05
**Portée :** `src/linalg/*`, `src/sp/*`, `src/autodiff/*`, `src/io/*`, `Cargo.toml`

---

## Modèle mental : cœur permanent vs sondes temporaires

La crate contient deux populations de code, de durées de vie différentes :

- **Le cœur** : `linalg/`, `sp/`, `scalar` : no_std pour toujours, c'est le produit. Il est
  aujourd'hui propre (aucune des 47 erreurs de build no_std n'en vient).
- **Les sondes** : `io/npy`, `io/load_inputs`, `main`, la dépendance `regex`, et *demain
  `HeapStorage`* : instrumentation de test et de benchmark. Elles existent pour mesurer la montée
  en charge sur des tenseurs que la pile ne peut pas porter, et seront débranchées progressivement
  jusqu'à ne laisser que du no_std partout.

Cette proposition ne cherche pas à rendre les sondes no_std. Elle cherche à ce que **le cœur reste
mesurable en no_std pendant que les sondes existent**, et à ce que les débrancher soit un
changement de features, pas une réécriture.

## Constat préalable : la frontière n'était pas matérialisée

> **Résolu.** État avant la Phase 0, conservé pour mémoire. Le job CI `embedded` cassait sur ce
> point ; il est vert depuis.

```
cargo build --no-default-features --lib      → 47 erreurs, toutes dans src/io/ (println!, format!, vec!, std::fs)
cargo build --target thumbv7em-none-eabihf   → échoue avant même notre code :
                                               regex-syntax + memchr → "can't find crate for `std`"
```

Aucune de ces erreurs n'est un défaut du cœur : elles disent seulement que les sondes sont
compilées inconditionnellement. La conséquence pratique est double, et c'est ce qui rend la Phase 0
utile plutôt que cosmétique :

1. **Pendant la migration**, il n'existe aucune commande qui réponde « le cœur est-il toujours
   no_std ? ». On migre `Vector`, `Matrix` et quatre tenseurs à l'aveugle, et on découvre les
   régressions à la fin.
2. **À la fin**, débrancher les sondes devient un big-bang au lieu d'un interrupteur. Si la config
   de build qui les exclut existe et tourne dès aujourd'hui, le débranchement progressif est déjà
   testé à chaque commit ; sinon c'est une opération à faire une fois, en aveugle, tout à la fin.

## Contexte

Tout buffer de la crate est un array inline : `Tensor{,3D,4D,6D}.data: [Scalar; NUMEL]`,
`Vector.data: [Scalar; N]`, `Matrix.data: [[Scalar; COLS]; ROWS]`, et les scratch de
`GradChecker::check` (`[Scalar; N]` × 3). Tout vit donc sur la pile dès que la valeur est possédée.
Le pipeline `.npy` overflow à `1x3x720x720`, mais c'est un symptôme : la même limite latente existe
partout ailleurs.

Décision : le lieu de stockage devient un paramètre permanent de la lib, pas un correctif ponctuel.
Défaut = pile (aucun call site existant ne change, aucune dépendance à `alloc`), heap disponible à
l'instanciation.

## Objectif

Un seul mécanisme de stockage, appliqué uniformément à tous les conteneurs, **sans toucher à la
logique de calcul** (`get`, `set`, `tensordot_*`, `im2col_view`, `mul_vec`, `matmul_accumulate`,
opérateurs). Ces corps doivent rester identiques au caractère près : seules les signatures et les
bornes changent. Toute méthode qui exige un changement de logique signale une fuite de
l'abstraction.

---

## Phase 0 : Matérialiser la frontière sondes / cœur ✅ (appliquée)

Objectif : une commande qui compile **le cœur seul**, sans sonde, et qui tourne en CI dès
maintenant. C'est à la fois le harnais de vérification des phases suivantes et le mode cible du
projet une fois les sondes retirées, atteint par soustraction de features, pas par réécriture.

1. ✅ `Cargo.toml` : `regex = { version = "1.13.1", optional = true }`. Features `alloc = []`,
   `std = ["alloc", "dep:regex"]`, `default = ["std"]`.
2. ✅ `lib.rs` : `#[cfg(feature = "std")] pub mod io;` ; le module `io` est intrinsèquement hosted
   (fs, chemins), il n'a pas à exister sur cible bare-metal.
3. ✅ `Cargo.toml` : `[[bin]] name = "ferrite", required-features = ["std"]` pour que
   `--no-default-features` ne tente pas de compiler le binaire de benchmark.
4. ✅ Job CI `embedded` étendu : thumbv7em nu, thumbv7em + `alloc`, et build no_std hôte.

Aucune ligne de `linalg/`, `sp/` ou `autodiff/` n'a été touchée : le cœur était déjà propre.

Cible : `cargo build --target thumbv7em-none-eabihf --no-default-features --lib` passe. C'est le
seul test qui prouve quoi que ce soit, et, à terme, la seule configuration qui restera.

Note : la feature `alloc` (donc `HeapStorage`) est côté sonde. Elle sert à porter les gros tenseurs
de benchmark ; le mode cible est `--no-default-features` tout court, où `HeapStorage` n'existe pas.
C'est précisément ce que la généricité de stockage achète : le cœur est écrit contre `Storage`, pas
contre `Box`, donc retirer `alloc` retire la sonde sans toucher une ligne de `linalg/`.

---

## Phase 1 : Le socle : `Buffer` + `Storage` ✅ (appliquée)

> Implémentée dans `src/linalg/storage.rs`, conforme au design ci-dessous.
> `ArenaStorage` n'est pas écrit ; seul le découpage `Storage` / `OwnedStorage` qui le rendra
> possible l'est. Le chargement passe par `Tensor4D::load_vec` (copie tas→tas via `as_flat_mut`)
> plutôt que par un `from_vec` dans le trait : ça garde `Vec` hors de l'abstraction, au prix d'un
> memcpy négligeable devant la contraction.

Divergence assumée par rapport à l'idée initiale : **paramétrer `Storage` par le type de buffer,
pas par `NUMEL`**. Un `Storage<const NUMEL: usize>` ne couvre que les buffers plats : `Matrix` avec
son `[[Scalar; COLS]; ROWS]` resterait hors du système, ou obligerait à aplatir `Matrix` en
`Matrix<R, C, NUMEL, S>` (rupture de tous les call sites, y compris `autodiff`). Générique sur le
buffer, un seul trait couvre tout.

Nouveau module `src/linalg/storage.rs` :

```rust
/// Un buffer de scalaires contigu, de taille connue à la compilation.
/// # Safety : impl réservée aux types dont le motif tout-à-zéro est valide
/// et dont la représentation est un bloc contigu de `LEN` scalaires.
pub unsafe trait Buffer: Sized {
    const LEN: usize;
    fn zeroed_inline() -> Self;              // construit sur place (pile)
    fn as_flat(&self) -> &[Scalar];
    fn as_flat_mut(&mut self) -> &mut [Scalar];
}

unsafe impl Buffer for Scalar { ... }                        // cas de base
unsafe impl<B: Buffer, const N: usize> Buffer for [B; N] { } // couvre [Scalar; N] ET [[Scalar; C]; R], tous rangs

/// Où vivent les éléments. N'impose rien sur la propriété.
pub trait Storage<B: Buffer>: Deref<Target = B> + DerefMut<Target = B> + Sized {}

/// Stockage possédé, donc constructible ex nihilo.
pub trait OwnedStorage<B: Buffer>: Storage<B> {
    fn zeroed() -> Self;
}
```

Le **split `Storage` / `OwnedStorage`** est le deuxième écart délibéré : `new()` exige
`OwnedStorage`, tandis que `get`/`set`/`view`/`im2col_view`/les contractions ne demandent que
`Storage`. Ça laisse la porte ouverte à un troisième stockage qui est le vrai besoin embarqué : un
buffer emprunté placé en `.bss` ou en SDRAM externe, sans allocateur :

```rust
pub struct StackStorage<B: Buffer>(B);             // toujours dispo, aucun alloc
pub struct HeapStorage<B: Buffer>(Box<B>);         // #[cfg(feature = "alloc")]
pub struct ArenaStorage<'a, B: Buffer>(&'a mut B); // Storage mais pas OwnedStorage
```

`ArenaStorage` peut n'être qu'esquissé (ou reporté), mais si `zeroed()` est dans le trait
principal, il devient impossible plus tard sans re-migration. Le coût de la prévoyance ici est nul.

**Point critique : `HeapStorage::zeroed()`** ne doit jamais matérialiser `B` sur la pile, sinon
l'overflow qu'on corrige revient par la bande. Le truc `vec![0.0; NUMEL].into_boxed_slice().try_into()`
ne marche que pour les buffers plats. La version générique :

```rust
let ptr = alloc::alloc::alloc_zeroed(Layout::new::<B>()) as *mut B;  // ~5 lignes d'unsafe
```

isolée dans ce module, justifiée par l'invariant `Buffer` ci-dessus, et valable pour tout rang.
C'est le seul `unsafe` de la migration. `Clone` sur `HeapStorage` doit passer par le même chemin
(allouer puis copier heap→heap, jamais `Box::new(*self.0)`).

**`Copy`/`Clone`** : `#[derive(Clone, Copy)]` sur un type générique en `S` engendre automatiquement
la borne `S: Copy`. Donc `StackStorage: Copy` ⇒ les instanciations pile restent `Copy` exactement
comme aujourd'hui, et les instanciations heap sont `Clone` seulement. Rien à écrire à la main, mais
c'est ce qui interdit aux tenseurs heap de traverser l'API `autodiff` actuelle (cf. Phase 4).

**Chargement des données.** `load_data(data: [Scalar; NUMEL])` prend l'array *par valeur* : le
temporaire est sur la pile du côté appelant, même si le tenseur est heap-backé. Quatre portes, par
ordre de coût, implémentées identiquement sur les quatre rangs de tenseur :

| méthode | dispo | coût |
|---|---|---|
| `new(data: [Scalar; NUMEL])` / `load_data([Scalar; NUMEL])` | partout | `new` construit et charge en un appel (pas de `mut` requis côté appelant) ; `load_data` conservée telle quelle pour recharger un tenseur existant. Pour les petits tenseurs (doctest de `correlate.rs`, `tests/`) |
| `load_slice(&[Scalar]) -> Result<(), LenMismatch>` | no_std, sans alloc | copie, pas de temporaire géant : la porte pour tout ce qui est gros sans allocateur |
| `from_vec(Vec<Scalar>) -> Result<Self, Vec<Scalar>>` | `alloc`, buffers plats | implémenté via `zeroed()` (alloc direct, jamais de temporaire pile) + copie tas→tas : la porte du pipeline `.npy`, remplace le `zeroed()`+`load_vec()` en deux étapes |

Ne pas supprimer `load_data` : ça casserait le doctest de `cross_correlate2d` et les tests pour
zéro gain.

---

## Phase 2 : Les tenseurs ✅ (appliquée aux quatre rangs)

> Fait : `Tensor` (2D), `Tensor3D`, `Tensor4D` (+ alias `Tensor4DBoxed`), `Tensor6D`,
> `tensordot_1`/`tensordot_2`/`tensordot_3`, `cross_correlate2d`. `tensordot_1`/`tensordot_2` et les
> méthodes qui produisent une forme différente de `Self` (`multiply`, `get_col`, `identity`...)
> restent en stockage par défaut (pas de paramètre de storage propagé sur leurs opérandes/sortie) ;
> rien dans le pipeline actuel n'en a besoin, cf. discussion initiale ci-dessus.
>
> Confirmation à l'usage : la friction anticipée sur l'inférence n'a pas eu lieu. Aucun call site
> existant n'a eu besoin d'un turbofish, l'annotation du résultat suffit partout. Et aucun corps de
> `get`/`set`/`tensordot_3`/`im2col_view` n'a changé, seulement les signatures, plus
> `&self.data` → `self.data.as_flat()` comme prévu.
>
> **Écart par rapport au sketch initial : `new()` s'est scindé en deux.** Le zéro-init générique
> (`S::zeroed()` sous borne `OwnedStorage`) est resté, mais renommé `zeroed()` et passé
> `pub(crate)` : invisible hors du crate, y compris depuis `tests/*.rs` (qui compile comme un crate
> externe). `new()` prend maintenant les données directement (`new(data: [Scalar; NUMEL])`,
> lui-même `zeroed()` + `load_data()` en une étape) : impossible pour un appelant externe d'obtenir
> un tenseur zéro silencieux en oubliant l'étape de chargement, puisqu'il n'y a plus de constructeur
> public qui ne prenne pas de données. Les algorithmes internes du crate (`identity`, `multiply`,
> `qr_decomposition`, `tensordot_*`, `sp::kernels::*`...), qui construisent un résultat élément par
> élément et ne peuvent pas fournir les données d'un coup, utilisent `zeroed()` directement ; ils
> sont dans le crate, donc `pub(crate)` ne les gêne pas.

Traitement mécanique et identique pour `Tensor`, `Tensor3D`, `Tensor4D`, `Tensor6D` :

- ajouter `S: Storage<[Scalar; NUMEL]> = StackStorage<[Scalar; NUMEL]>` **en dernier** paramètre,
  `data: S` ;
- `new()` → `S::zeroed()` sous borne `S: OwnedStorage<…>` ;
- `view()` / `im2col_view()` : `&self.data` → `self.data.as_flat()` (les vues `TensorView` /
  `TensorView6D` sont déjà `&'a [Scalar]`, elles ne changent pas d'un caractère) ;
- `Tensor4D::get_data()` : `&[f32; NUMEL]` → `&[Scalar]` via `as_flat()`. Seul consommateur :
  `write_npy(…, &[f32])`, qui accepte déjà un slice ;
- `impl Rank6 for Tensor6D<…>` : ajouter `S` et le propager.

**Storage du résultat des contractions** : paramètre générique sur `tensordot_1`, `tensordot_2`,
`tensordot_3` et `cross_correlate2d`. Sinon le résultat `1x718x718x2` (1 031 048 f32 ≈ 4 Mo)
overflow alors même que les opérandes sont sur le tas. C'est le cas le plus gros du pipeline :
l'option « documenter que le résultat reste pile » ne tient pas.

**Friction à connaître d'avance** : Rust n'autorise pas de valeur par défaut sur un paramètre
générique de *fonction*. `tensordot_3` ne peut donc pas hériter du défaut `StackStorage` ; `SC` est
déduit du contexte. En pratique ça passe sans rien changer, parce que tous les call sites actuels
annotent le résultat (`let result: Tensor4D<1, 62, 62, 2, 7688> = tensordot_3(…)`,
`-> Tensor4D<N, H_OUT, W_OUT, K, NUMEL_Y>` dans `cross_correlate2d`) et l'annotation fournit le
défaut du *type*. Les appels non annotés, s'il en apparaît, exigeront un turbofish. À vérifier en
premier sur un fichier avant de dérouler les quatre types.

---

## Phase 3 : `Vector` et `Matrix`

- `Vector<N, S = StackStorage<[Scalar; N]>>`. `new([Scalar; N])` reste (constructeur ergonomique,
  utilisé partout dans `autodiff` et les tests) ; ajouter `from_slice`. Les opérateurs (`Add`,
  `Sub`, `Mul<Scalar>`, `Neg`, `Div`, `hadamard`, `orthogonal_projection`) construisent tous un
  `[0.0; N]` temporaire puis `Vector::new(data)` : ils deviennent `S::zeroed()` + écritures
  indexées. Sortie = storage de l'opérande gauche ; opérandes de storages différents autorisés
  (`impl Add<&Vector<N, S2>> for &Vector<N, S1> → Vector<N, S1>`). Les impls par valeur délèguent
  déjà aux impls par référence, elles ne demandent qu'une borne `S: Clone`.
- `Matrix<ROWS, COLS, S = StackStorage<[[Scalar; COLS]; ROWS]>>`. Grâce au `Buffer` générique sur
  les arrays imbriqués, **aucun aplatissement, aucun `NUMEL` à ajouter, aucun call site à toucher**.
  `self.data[i][j]` continue de fonctionner via `Deref`. `get_col` / `mul_vec` / `scale` /
  `transpose` construisent leur sortie via `S::zeroed()` au lieu de `[0.0; ROWS]`.
- `decomposition.rs` (250 lignes, `gram_schmidt`, `qr`, `svd`, `solve_*`) : à relire, mais si les
  signatures sont exprimées en `Matrix<R, C>` / `Vector<N>` sans mention du buffer, les défauts les
  laissent intactes. Ne les généraliser que si un besoin se manifeste : une factorisation QR sur
  cible edge ne se fait pas sur du 720p.

---

## Phase 4 : `autodiff` : ce qu'on fait, ce qu'on ne fait pas

**Hors scope, explicitement** : propager `S` dans `Linear<IN, OUT>` / `LinearGrads` et les traits
`Module` / `Params` / `FlatGrads` / `Update`. Ça ajouterait deux paramètres de storage par couche
(poids + biais) pour un cas d'usage, inférence edge sur petites couches, qui est exactement celui
où la pile est le bon choix. Coût en bruit très supérieur au bénéfice.

**Dans le scope**, deux corrections qui relèvent du même problème sans avoir besoin de la
machinerie :

1. `GradChecker::check::<N>` alloue **trois** `[Scalar; N]` sur la pile (`buf_ana`, `buf_num`,
   `errors`), N = nombre de paramètres du réseau. Même bombe à retardement, en pire (×3).
   Correctif recommandé : faire prendre les scratch à l'appelant (`&mut [Scalar]`) plutôt que de
   storage-ifier `GradChecker` : ça marche en no_std sans `alloc`, sans trait, et supprime au
   passage le `debug_assert_eq!(net.num_params(), N)` qui ne vérifie qu'en debug.
2. Les bornes `Input: Copy` et `<Net as Module<Input>>::Output: Copy` dans `train_step` et `check` :
   à relâcher en `Clone` maintenant. Un jour où un tenseur heap-backé traverse un réseau, elles
   bloquent ; les changer coûte une ligne aujourd'hui et une seconde migration plus tard.

---

## Phase 5 : Pipeline `io` / bench ✅ (points 1 et 2 appliqués)

> Les 20 bras du `match` sont factorisés en une macro `bench_case!` et tous rebranchés (les 15 qui
> étaient commentés parce qu'ils overflowaient tournent maintenant). Points 3 et 4 non traités :
> hors zone mesurée pour le premier, non bloquant pour le second.

1. `load_inputs.rs` : `Tensor4DBoxed` pour `vid_tensor` / `fil_tensor` / `result`, chargement via
   `from_vec` (un seul appel, sans `mut` côté appelant : copie tas→tas en interne, cf. Phase 1),
   `unwrap_or_else(|_| panic!("…"))` plutôt que `.expect()`, pour ne pas déverser un `Vec<Scalar>`
   de 1,5 M d'éléments dans un message de panique.
2. **Avant** de migrer, factoriser le `match` : ~20 bras quasi identiques (dont 15 commentés parce
   qu'ils overflow). Un
   `macro_rules! bench_case!(vid, fil, key, N,C,H,W,NUMEL_X, K,NUMEL_F, H_OUT,W_OUT,NUMEL_Y)`
   réduit le diff de la migration de 20 endroits à 1, et permet de décommenter tous les cas d'un
   coup une fois le heap en place. Fait dans cet ordre, la migration devient triviale ; fait après,
   c'est 20 éditions à la main.
3. `write_npy` écrit **un `write_all` par f32** sur un `File` non bufferisé, ~1 M d'appels système
   pour le cas 720p. **Hors zone mesurée** : la mesure porte sur `tensordot_3` seul, donc ça
   n'affecte pas la justesse des chiffres, seulement le temps d'attente entre deux itérations de
   bench. Un `BufWriter` est une ligne si le confort le justifie ; sinon, non-sujet.
4. `read_npy` : `Vec<u8>` puis `Vec<f32>` = deux copies complètes (18 Mo transitoires pour du 720p),
   et **aucune vérification** que `data.len() == shape.iter().product()`. C'est ce contrôle manquant
   qui fait que l'erreur remonte tard, sous forme de panique dans `load_data`. Le déplacer dans
   `read_npy`, où le message peut être utile.

---

## Points à trancher

| # | Question | Recommandation |
|---|---|---|
| 1 | `Storage<B: Buffer>` vs `Storage<const NUMEL>` | **Buffer** : sinon `Matrix` reste hors du système ou doit être aplati (rupture de tous les call sites) |
| 2 | `alloc_zeroed` (unsafe) vs `vec![].into_boxed_slice()` | **`alloc_zeroed`** : le truc `vec!` ne couvre pas les buffers imbriqués ; ~5 lignes d'unsafe isolées valent mieux qu'une abstraction qui ne s'applique qu'à la moitié des types |
| 3 | `Storage` + `OwnedStorage` séparés, ou un seul trait avec `zeroed()` | **Séparés** : coût nul maintenant, seule façon d'accueillir plus tard un stockage emprunté (`.bss`, SDRAM externe), qui est le vrai besoin embarqué au-delà du binôme pile/tas |
| 4 | Propager `S` dans `autodiff` | **Non** (Phase 4), mais relâcher `Copy` → `Clone` tout de suite |
| 5 | Nom de la feature | `alloc`, impliquée par `std` : `heap` décrirait le cas d'usage, `alloc` décrit la dépendance réelle et suit la convention de l'écosystème |

## Matrice de vérification

```
cargo build --no-default-features --lib                                  # no_std strict, sans alloc
cargo build --no-default-features --features alloc --lib                 # no_std + heap
cargo build                                                              # std (défaut)
cargo build --target thumbv7em-none-eabihf --no-default-features --lib   # la seule preuve qui compte
cargo test                                                               # dont les doctests de correlate.rs
```

Plus :

- le cas `1x3x720x720` / `2x3x3x3` (celui qui overflow) passe avec `Tensor4DBoxed` ;
- un code existant écrivant `Tensor4D::<1,3,32,32,3072>::new()` sans storage explicite compile
  inchangé et tourne toujours sur la pile ;
- **revue du diff** : aucun corps de `get` / `set` / `tensordot_*` / `im2col_view` / `mul_vec` /
  `matmul_accumulate` modifié autrement qu'en signature ou en bornes. Une seule exception attendue
  et acceptée : `&self.data` → `self.data.as_flat()` dans `view` et `im2col_view`.

---

## Ordre d'exécution

**Phase 0 avant tout le reste**, non parce que le cœur serait cassé (il ne l'est pas), mais parce
qu'elle produit le harnais qui rend les phases 1 à 3 vérifiables au fur et à mesure, et parce que
la config de build qu'elle installe *est* l'état final du projet une fois les sondes débranchées.
La construire maintenant, c'est tester le débranchement à chaque commit au lieu d'une seule fois à
la fin.

## Débranchement des sondes (état final visé)

Ordre de retrait, du plus tardif au plus précoce dans la vie du projet :

| sonde | retirée quand | mécanisme |
|---|---|---|
| `io/load_inputs`, `io/npy`, `main`, `regex` | les campagnes de bench sur données réelles sont terminées | `--no-default-features` les exclut déjà (Phase 0) |
| `HeapStorage`, feature `alloc` | plus besoin de tenseurs dépassant la pile | `--no-default-features` les exclut déjà (Phase 1) |
| `GradChecker` scratch buffers | jamais : l'API `&mut [Scalar]` de la Phase 4 est déjà no_std sans `alloc` | n/a |

Reste alors : `linalg/` + `sp/` + `scalar`, en `StackStorage` (et `ArenaStorage` si la Phase 1 le
concrétise), sans allocateur.
