use log::{info, debug, trace};
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
}

#[derive(Debug, Deserialize)]
struct Comp {
    members: HashMap<String, Class>,
}

#[derive(Debug, Deserialize)]
struct Settings {
    #[serde(flatten)]
    comp: Comp,
    clears_per_week: Vec<TierLevel>,
    num_extra_vault_items: usize,
    trading_rule: TradingRule,
    num_samples: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct TradingRule {
    target: TradingTargetRule,
}

#[derive(Deserialize, Debug, Copy, Clone)]
enum TradingTargetRule {
    MostPieces,
    LeastPieces,
    Arbitrary,
}

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum TierLevel {
    AnyLevel,
    Lfr,
    Normal,
    Heroic,
    Mythic,
}

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
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

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
enum Token {
    Conqueror,
    Vanquisher,
    Protector,
}

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

#[derive(Deserialize, Debug, Copy, Clone, Ord, PartialEq, PartialOrd, Eq)]
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
    comp: Vec<Token>,
    has: BTreeMap<Slot, Vec<Option<TierLevel>>>,
    num_slots: Vec<u8>,
    completion_week: Vec<Option<u8>>,
    trading_rule: TradingRule,
    bonus_chance: Bernoulli,
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
    F: Fn(u8, Option<u8>) -> Trade,
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

        let trade = condition(items, target_items);
        if trade == Trade::Yes || trade == Trade::CoinFlip && rng.gen_bool(0.5) {
            target = Some(ix);
            target_items = Some(items);
        }
    }

    // if everyone has it, give it to the person with the lowest ilvl of it
    if target.is_none() {
        debug!("All eligible trade targets have {slot:?}. Maximizing ilvl gap");
        for (ix, &target_token) in state.comp.iter().enumerate() {
            if target_token != token || state.has(ix, slot, level) {
            continue;
        }
            let ix_level = state.has[&slot][ix];
            let items = state.num_slots[ix];

            if ix_level < target_level || ix_level == target_level && rng.gen_bool(0.5) {
                target = Some(ix);
                target_level = ix_level;
                target_items = Some(items);
            }
        }
    }

    debug!("Trading {level:?} {slot:?} to {target:?}, who has {target_level:?} (items: {target_items:?})");

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

    fn store_tier(&mut self, ix: usize, slot: Slot, level: TierLevel) {
        if !self.has(ix, slot, level) {
            self.has.get_mut(&slot).unwrap()[ix] = Some(level);
            self.num_slots[ix] += 1;
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
            MostPieces => {
                trade_to(&self, token, slot, level, |items, target_items| {
                    let target_items = target_items.unwrap_or(0);
                    if items > 4 {
                        Trade::No
                    } else if items > target_items {
                        Trade::Yes
                    } else if items == target_items {
                        Trade::CoinFlip
                    } else {
                        Trade::No
                    }
                })
            }
            LeastPieces => {
                trade_to(&self, token, slot, level, |items, target_items| {
                    if items < target_items.unwrap_or(5) {
                        Trade::Yes
                    } else if items == target_items.unwrap_or(5) {
                        Trade::CoinFlip
                    } else {
                        Trade::No
                    }
                })
            }
        }
    }

    fn award_tier<R: Rng>(&mut self, rng: &mut R, slot: Slot, level: TierLevel) {
        let n = self.num_items(rng);
        let awardees = self
            .comp
            .iter()
            .cloned()
            .enumerate()
            .choose_multiple(rng, n);

        for (ix, token) in awardees {
            if let Some(other) = self.trade_item(ix, slot, token, level) {
                self.store_tier(other, slot, level);
            } else {
                self.store_tier(ix, slot, level);
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

            for &slot in raid_slots.chain(bonus_slots) {
                if !self.has(ix, slot, TierLevel::Heroic) {
                    self.store_tier(ix, slot, TierLevel::Heroic);
                    break;
                }
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
}

fn sample_completion_time(settings: &Settings) -> Sample {
    let mut weeks = 0;
    let mut state = State::from_settings(settings);
    let mut rng = rand::thread_rng();

    let mut pct_2pc = Vec::<f64>::new();
    let mut pct_4pc = Vec::<f64>::new();

    loop {
        weeks += 1;
        for &level in &settings.clears_per_week {
            state.award_tier(&mut rng, Slot::Helm, level);
            state.award_tier(&mut rng, Slot::Legs, level);
            state.award_tier(&mut rng, Slot::Gloves, level);
            if weeks > 1 {
                state.award_tier(&mut rng, Slot::Shoulders, level);
                state.award_tier(&mut rng, Slot::Chest, level);
            }
        }

        state.award_vault(&mut rng, weeks > 1);
        state.mark_completion_dates(weeks);

        pct_2pc.push(state.completion_pct(2));

        let completion = state.completion_pct(4);
        pct_4pc.push(completion);

        if completion >= 1.0 {
            break;
        }
    }

    Sample {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    femme::start();

    let opts = Opts::from_args();
    let settings_raw = std::fs::read(&opts.config)?;
    let settings = toml::from_slice::<Settings>(&settings_raw)?;

    let mut samples = Vec::with_capacity(settings.num_samples);

    (0..settings.num_samples)
        .into_par_iter()
        .map(|_ix| sample_completion_time(&settings))
        .collect_into_vec(&mut samples);

    std::fs::write(&opts.output, serde_json::to_vec(&samples)?)?;

    let mut weeks = Data::new(samples.iter().map(|s| s.n_weeks as f64).collect::<Vec<_>>());

    println!(
        "Overall Completion: {} - {} weeks (Avg {}) (Trade to {:?})",
        weeks.quantile(0.025),
        weeks.quantile(0.975),
        weeks.iter().mean(),
        settings.trading_rule.target
    );

    print_per_player(&settings, &samples);

    Ok(())
}
