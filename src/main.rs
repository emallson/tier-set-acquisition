use log::debug;
use rand::distributions::{Bernoulli, Distribution};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::Binomial;
use statrs::statistics::{Data, OrderStatistics, Statistics};
use std::collections::{BTreeMap, HashMap};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opts {
    config: String,
    output: String,
    #[structopt(long)]
    debug: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct Comp {
    members: HashMap<String, Class>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Settings {
    #[serde(flatten)]
    comp: Comp,
    clears_per_week: Vec<TierLevel>,
    mythic_kills: Vec<u8>,
    mythic_first_slot: Slot,
    num_extra_vault_items: usize,
    trading_rule: TradingRule,
    num_samples: usize,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
struct TradingRule {
    target: TradingTargetRule,
}

#[derive(Deserialize, Debug, Copy, Clone, Serialize)]
enum TradingTargetRule {
    MostPieces,
    LeastPieces,
    Arbitrary,
    MostSetCompletionsMostPieces,
    MostSetCompletionsLeastPieces,
}

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum TierLevel {
    AnyLevel,
    Lfr,
    Normal,
    Heroic,
    Mythic,
}

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq, Serialize)]
enum Class {
    DeathKnight,
    DemonHunter,
    Druid,
    Hunter,
    Mage,
    Monk,
    Paladin,
    Priest,
    Rogue,
    Shaman,
    Warlock,
    Warrior,
}

#[cfg(not(feature = "old_tokens"))]
#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq, Serialize)]
enum Token {
    Mystic,
    Venerated,
    Zenith,
    Dreadful
}

#[cfg(not(feature = "old_tokens"))]
impl Class {
    fn into_token(self) -> Token {
        use Class::*;
        match self {
            Druid | Hunter | Mage => Token::Mystic,
            Paladin | Priest | Shaman => Token::Venerated,
            Monk | Rogue | Warrior => Token::Zenith,
            DemonHunter | DeathKnight | Warlock => Token::Dreadful,
        }
    }
}

#[cfg(feature = "old_tokens")]
#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq, Serialize)]
enum Token {
    Conqueror,
    Vanquisher,
    Protector,
}

#[cfg(feature = "old_tokens")]
impl Class {
    fn into_token(self) -> Token {
        use Class::*;
        match self {
            DemonHunter | Paladin | Priest | Warlock => Token::Conqueror,
            DeathKnight | Druid | Mage | Rogue => Token::Vanquisher,
            Hunter | Monk | Shaman | Warrior => Token::Protector,
        }
    }
}

#[derive(Deserialize, Debug, Copy, Clone, Ord, PartialEq, PartialOrd, Eq, Serialize)]
enum Slot {
    Helm,
    Shoulders,
    Chest,
    Gloves,
    Legs,
}

const SLOTS: [Slot; 5] = [
    Slot::Helm,
    Slot::Shoulders,
    Slot::Chest,
    Slot::Gloves,
    Slot::Legs,
];

#[derive(Debug)]
struct State {
    record_waste: bool,
    comp: Vec<Token>,
    has: BTreeMap<Slot, Vec<Option<TierLevel>>>,
    num_slots: Vec<u8>,
    completion_week: Vec<Option<u8>>,
    trading_rule: TradingRule,
    bonus_chance: Bernoulli,
    wasted_vaults: usize,
    wasted_drops: usize,
    /// Chance for a vault slot to be tier for non-raid slots. Raid slots use the raid loot table.
    nonraid_vault_chance: Binomial,
}

#[derive(Debug, PartialEq, Eq)]
enum Trade {
    Yes,
    No,
    CoinFlip,
}

fn trade_to<F>(
    state: &State,
    token: Token,
    slot: Slot,
    level: TierLevel,
    condition: F,
) -> Option<usize>
where
    F: Fn(u8, u8) -> Trade,
{
    let mut rng = rand::thread_rng();
    let mut target = None;
    let mut target_items = None;
    let mut target_level = None;

    debug!("Searching for trade target for {level:?} {slot:?}");

    // Give it to the person with the `condition` pieces among those without it.
    for (ix, &target_token) in state.comp.iter().enumerate() {
        if target_token != token || state.has(ix, slot, TierLevel::AnyLevel) {
            continue;
        }
        let items = state.num_slots[ix];

        if target.is_none() {
            target = Some(ix);
            target_items = Some(items);
            continue;
        }

        let trade = condition(items, target_items.unwrap());
        if trade == Trade::Yes || trade == Trade::CoinFlip && rng.gen_bool(0.5) {
            target = Some(ix);
            target_items = Some(items);
        }
    }

    // if everyone has it, give it to the person with the lowest ilvl of it
    if target.is_none() {
        debug!("All eligible trade targets have {slot:?}. Maximizing ilvl gap");
        for (ix, &target_token) in state.comp.iter().enumerate() {
            if target_token != token { continue; }
            if state.has(ix, slot, level) {
                debug!("Not trading to {ix}. They have at least{level:?} in {slot:?}.");
                continue;
            }
            let ix_level = state.has[&slot][ix];
            let items = state.num_slots[ix];

            if target.is_none() || ix_level < target_level || ix_level == target_level && rng.gen_bool(0.5) {
                debug!("Marking {ix} for possible trade. They have {ix_level:?} in {slot:?}.");
                target = Some(ix);
                target_level = ix_level;
                target_items = Some(items);
            } else {
                debug!("Not trading to {ix}. They have {ix_level:?} in {slot:?}.");
            }
        }
    }

    debug!("Trading {level:?} {token:?} {slot:?} to {target:?}, who has {target_level:?} (items: {target_items:?})");

    target
}

impl State {
    fn num_items<R: Rng>(&self, rng: &mut R) -> usize {
        let base = self.comp.len() / 10;
        let bonus = self.bonus_chance.sample(rng);

        base + bonus as usize
    }

    fn has(&self, ix: usize, slot: Slot, level: TierLevel) -> bool {
        self.has[&slot][ix]
            .map(|target| target >= level)
            .unwrap_or(false)
    }

    fn store_tier(&mut self, ix: usize, slot: Slot, level: TierLevel) -> bool {
        if !self.has(ix, slot, level) {
            let v = self.has.get_mut(&slot).unwrap();
            let old = v[ix];
            v[ix] = Some(level);
            if old.is_none() {
                self.num_slots[ix] += 1;
            }
            debug!("Item {level:?} {slot:?} awarded to {ix}");
            true
        } else {
            debug!("Drop {level:?} {slot:?} awarded to {ix} WASTED");
            false
        }
    }

    fn mark_completion_dates(&mut self, week: u8) {
        for (&slots, compl) in self.num_slots.iter().zip(self.completion_week.iter_mut()) {
            if slots >= 4 && compl.is_none() {
                *compl = Some(week);
            }
        }
    }

    fn trade_item(
        &self,
        source: usize,
        slot: Slot,
        token: Token,
        level: TierLevel,
    ) -> Option<usize> {
        if !self.has(source, slot, level) {
            // per comments from Lore, we can't trade items that we don't have. this means that if
            // we just got a helm, we can't trade it even if we have same ilvl helm from M+
            return None;
        }

        // okay, so we're trading
        use TradingTargetRule::*;
        match self.trading_rule.target {
            Arbitrary => self
                .comp
                .iter()
                .enumerate()
                .filter(|&(ix, &target_token)| target_token == token && !self.has(ix, slot, level))
                .nth(0)
                .map(|(ix, _)| ix),
            MostSetCompletionsMostPieces => trade_to(&self, token, slot, level, |items, target_items| {
                if items == target_items {
                    // randomize equal options to break ordering dependency in results
                    Trade::CoinFlip
                } else if target_items == 3 {
                    // if the current target would get a 4pc, don't switch
                    Trade::No
                } else if target_items == 1 && items != 3 {
                    // if the current target would get a 2pc and the new target would not get a
                    // 4pc, don't switch
                    Trade::No
                } else if items == 3 || items == 1 {
                    // otherwise complete a set
                    Trade::Yes
                } else if items > 4 {
                    // don't give out 5pc
                    Trade::No
                } else if items > target_items {
                    Trade::Yes
                } else {
                    Trade::No
                }
            }),
            MostPieces => trade_to(&self, token, slot, level, |items, target_items| {
                if items > 4 {
                    Trade::No
                } else if items > target_items {
                    Trade::Yes
                } else if items == target_items {
                    Trade::CoinFlip
                } else {
                    Trade::No
                }
            }),
            MostSetCompletionsLeastPieces => trade_to(&self, token, slot, level, |items, target_items| {
                if items == target_items {
                    // randomize equal options to break ordering dependency in results
                    Trade::CoinFlip
                } else if target_items == 3 {
                    // if the current target would get a 4pc, don't switch
                    Trade::No
                } else if target_items == 1 && items != 3 {
                    // if the current target would get a 2pc and the new target would not get a
                    // 4pc, don't switch
                    Trade::No
                } else if items == 3 || items == 1 {
                    // otherwise complete a set
                    Trade::Yes
                } else if items < target_items {
                    Trade::Yes
                } else {
                    Trade::No
                }
            }),
            LeastPieces => trade_to(&self, token, slot, level, |items, target_items| {
                if items < target_items {
                    Trade::Yes
                } else if items == target_items {
                    Trade::CoinFlip
                } else {
                    Trade::No
                }
            }),
        }
    }

    fn award_tier<R: Rng>(&mut self, rng: &mut R, slot: Slot, level: TierLevel) {
        let n = if level == TierLevel::Mythic { 2 } else { self.num_items(rng) };
        let awardees = self
            .comp
            .iter()
            .cloned()
            .enumerate()
            .choose_multiple(rng, n);

        for (ix, token) in awardees {
            let not_wasted = if let Some(other) = self.trade_item(ix, slot, token, level) {
                self.store_tier(other, slot, level)
            } else {
                self.store_tier(ix, slot, level)
            };

            if !not_wasted && self.record_waste {
                self.wasted_drops += 1;
            }
        }
    }

    fn completion_pct(&self, bonus: u8) -> f64 {
        let n_players = self.comp.len() as f64;
        self.num_slots.iter().filter(|&&c| c >= bonus).count() as f64 / n_players
    }

    fn from_settings(settings: &Settings) -> State {
        let comp = settings
            .comp
            .members
            .values()
            .cloned()
            .map(Class::into_token)
            .collect::<Vec<_>>();
        let num_players = comp.len();
        let slots = vec![None; num_players];

        State {
            record_waste: true,
            comp,
            bonus_chance: Bernoulli::from_ratio((num_players % 10) as u32, 10).unwrap(),
            has: SLOTS
                .iter()
                .cloned()
                .map(|slot| (slot, slots.clone()))
                .collect(),
            completion_week: vec![None; num_players],
            num_slots: vec![0; num_players],
            trading_rule: settings.trading_rule.clone(),
            nonraid_vault_chance: Binomial::new(0.2, settings.num_extra_vault_items as u64)
                .unwrap(),
            wasted_drops: 0,
            wasted_vaults: 0,
        }
    }

    /// Calculate vault drops for each raider. Slightly inaccurate first week because it doesn't
    /// model the number of items that you can't get from the final 3 on raid row, only prevents you
    /// from getting inappropriate tier slots and from getting all 3 vault options on the raid row.
    ///
    /// Also assumes you can get those slots from M+/PvP rows first week, because why would
    /// blizzard be consistent.
    fn award_vault<R: Rng>(&mut self, rng: &mut R, full_raid: bool) {
        /// number of distinct items you can get from raid
        const RAID_ITEMS: f64 = 42.0;

        debug!("Awarding vault items (full raid?: {full_raid})");

        let raid_dist = Binomial::new(5.0 / RAID_ITEMS, if full_raid { 3 } else { 2 }).unwrap();
        for ix in 0..self.comp.len() {
            let n_tier_drops = raid_dist.sample(rng).round() as usize;
            let raid_slots = SLOTS
                .choose_multiple_weighted(rng, n_tier_drops, |&slot| {
                    if full_raid {
                        1.0
                    } else if slot == Slot::Shoulders || slot == Slot::Chest {
                        0.0
                    } else {
                        1.0
                    }
                })
                .unwrap();

            let bonus_drops = self.nonraid_vault_chance.sample(rng).round() as usize;
            let bonus_slots = SLOTS.choose_multiple(rng, bonus_drops);

            let mut loot_taken = false;
            let mut loot_options = false;
            let bonus_level = if full_raid { TierLevel::Mythic } else { TierLevel::Heroic };
            for &slot in bonus_slots {
                loot_options = true;
                if !self.has(ix, slot, bonus_level) {
                    self.store_tier(ix, slot, bonus_level);
                    loot_taken = true;
                    break;
                }
            }

            if !loot_taken {
                for &slot in raid_slots {
                    loot_options = true;
                    if !self.has(ix, slot, TierLevel::Heroic) {
                        self.store_tier(ix, slot, TierLevel::Heroic);
                        loot_taken = true;
                        break;
                    }
                }
            }

            if !loot_taken && loot_options && self.record_waste {
                debug!("Vault for {ix} WASTED");
                self.wasted_vaults += 1;
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct Sample {
    completion_weeks: Vec<u8>,
    n_weeks: u8,
    pct_2pc: Vec<f64>,
    pct_4pc: Vec<f64>,
    wasted_vaults: usize,
    wasted_drops: usize,
}

fn sample_completion_time(settings: &Settings) -> Sample {
    let mut weeks = 0u8;
    let mut state = State::from_settings(settings);
    let mut rng = rand::thread_rng();

    let mut pct_2pc = Vec::<f64>::new();
    let mut pct_4pc = Vec::<f64>::new();

    loop {
        weeks += 1;

        debug!("Processing week {weeks}");

        if weeks > 4 {
            state.record_waste = false;
        }

        if weeks > 1 {
            state.award_vault(&mut rng, weeks > 2);
        }

        for &level in &settings.clears_per_week {
            debug!("Awarding drops for {level:?}");
            state.award_tier(&mut rng, Slot::Helm, level);
            state.award_tier(&mut rng, Slot::Legs, level);
            state.award_tier(&mut rng, Slot::Gloves, level);
            if weeks > 1 {
                state.award_tier(&mut rng, Slot::Shoulders, level);
                state.award_tier(&mut rng, Slot::Chest, level);
            }
        }

        if weeks > 1 {
            debug!("Awarding drops for Mythic");
            let mythic_bosses = settings
                .mythic_kills
                .get((weeks - 1) as usize)
                .or(settings.mythic_kills.last())
                .cloned()
                .unwrap_or(0);

            if mythic_bosses >= 10 {
                state.award_tier(&mut rng, Slot::Chest, TierLevel::Mythic);
            }
            if mythic_bosses >= 9 {
                state.award_tier(&mut rng, Slot::Shoulders, TierLevel::Mythic);
            }
            if mythic_bosses >= 8 {
                state.award_tier(&mut rng, Slot::Helm, TierLevel::Mythic);
            }
            // special handling for the first 7 because you could go gloves *or* legs first
            if mythic_bosses >= 7 {
                state.award_tier(&mut rng, Slot::Legs, TierLevel::Mythic);
                state.award_tier(&mut rng, Slot::Gloves, TierLevel::Mythic);
            } else if mythic_bosses >= 4 {
                state.award_tier(&mut rng, settings.mythic_first_slot, TierLevel::Mythic);
            }
        }

        state.mark_completion_dates(weeks);

        pct_2pc.push(state.completion_pct(2));

        let completion = state.completion_pct(4);
        assert!(weeks > 1 || completion == 0.0, "4pc completion should be impossible week 1");
        pct_4pc.push(completion);

        if completion >= 1.0 {
            break;
        }
    }

    Sample {
        wasted_vaults: state.wasted_vaults,
        wasted_drops: state.wasted_drops,
        completion_weeks: state
            .completion_week
            .into_iter()
            .map(Option::unwrap)
            .collect(),
        n_weeks: weeks,
        pct_2pc,
        pct_4pc,
    }
}

fn print_per_player(settings: &Settings, samples: &[Sample]) {
    let mut completions: Vec<Vec<f64>> = Vec::with_capacity(settings.comp.members.len());
    for _ in 0..settings.comp.members.len() {
        completions.push(Vec::with_capacity(samples.len()));
    }

    for sample in samples {
        for (ix, &wk) in sample.completion_weeks.iter().enumerate() {
            completions[ix].push(wk as f64);
        }
    }

    let mut names = settings.comp.members.iter().collect::<Vec<_>>();
    names.sort_by_key(|t| t.0);
    for (ix, wks) in completions.into_iter().enumerate() {
        let mut data = Data::new(wks);
        println!(
            "{:>12}\t{:>12}\t{}\t{}",
            names[ix].0,
            format!("{:?}", names[ix].1),
            data.quantile(0.025),
            data.quantile(0.975)
        );
    }
}

#[derive(Serialize, Debug)]
struct SimOutput<'a> {
    settings: &'a Settings,
    players_ordered: &'a Vec<String>,
    samples: &'a Vec<Sample>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::from_args();

    femme::with_level(if opts.debug { femme::LevelFilter::Debug } else { femme::LevelFilter::Info });
    let settings_raw = std::fs::read(&opts.config)?;
    let settings = toml::from_slice::<Settings>(&settings_raw)?;
    let num_samples = if opts.debug { 1 } else { settings.num_samples };

    let mut samples = Vec::with_capacity(num_samples);

    (0..num_samples)
        .into_par_iter()
        .map(|_ix| sample_completion_time(&settings))
        .collect_into_vec(&mut samples);

    let names = settings.comp.members.keys().cloned().collect::<Vec<_>>();
    std::fs::write(&opts.output, serde_json::to_vec(&SimOutput { settings: &settings, players_ordered: &names, samples: &samples })?)?;

    let mut weeks = Data::new(samples.iter().map(|s| s.n_weeks as f64).collect::<Vec<_>>());
    let wasted_vaults = samples.iter().map(|s| s.wasted_vaults as f64).mean();
    let wasted_drops = samples.iter().map(|s| s.wasted_drops as f64).mean();

    println!(
        "Overall Completion: {} - {} weeks (Avg {}) (Trade to {:?}) (Avg Wasted Vaults: {wasted_vaults}, Avg Wasted Drops: {wasted_drops})",
        weeks.quantile(0.025),
        weeks.quantile(0.975),
        weeks.iter().mean(),
        settings.trading_rule.target
    );

    print_per_player(&settings, &samples);

    Ok(())
}
