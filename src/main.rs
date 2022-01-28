use bitvec::prelude::*;
use rand::distributions::{Bernoulli, Distribution};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use statrs::statistics::Statistics;
use statrs::distribution::Binomial;
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
    clears_per_week: usize,
    num_extra_vault_items: usize,
    trading_rule: TradingRule,
    num_samples: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct TradingRule {
    source: TradingSourceRule,
    target: TradingTargetRule,
}

/// The rule for trading tier to other players of the same class. Loot that has already been
/// acquired will always be traded if it has already been acquired---this determines what to do
/// with pieces that haven't yet been acquired.
#[derive(Deserialize, Debug, Copy, Clone)]
enum TradingSourceRule {
    NoTrading,
    Always,
    After2pc,
    After4pc,
    After5pc,
}

#[derive(Deserialize, Debug, Copy, Clone)]
enum TradingTargetRule {
    MostPieces,
    LeastPieces,
    Arbitrary,
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
    comp: Vec<Class>,
    has: BTreeMap<Slot, BitArray>,
    num_slots: Vec<u8>,
    trading_rule: TradingRule,
    bonus_chance: Bernoulli,
    /// Chance for a vault slot to be tier for non-raid slots. Raid slots use the raid loot table.
    nonraid_vault_chance: Binomial,
}

impl State {
    fn num_items<R: Rng>(&self, rng: &mut R) -> usize {
        let base = self.comp.len() / 5;
        let bonus = self.bonus_chance.sample(rng);

        base + bonus as usize
    }

    fn store_tier(&mut self, ix: usize, slot: Slot) {
        if !self.has[&slot][ix] {
            self.has.get_mut(&slot).unwrap().set(ix, true);
            self.num_slots[ix] += 1;
        }
    }

    fn trade_item(&self, source: usize, slot: Slot, cls: Class) -> Option<usize> {
        let num_pieces = self.num_slots[source];

        if !self.has[&slot][source] {
            use TradingSourceRule::*;
            match self.trading_rule.source {
                NoTrading => return None,
                Always => {},
                After2pc => {
                    if num_pieces < 2 {
                        return None;
                    }
                }
                After4pc => {
                    if num_pieces < 4 {
                        return None;
                    }
                }
                After5pc => {
                    if num_pieces < 5 {
                        return None;
                    }
                }
            }
        }

        // okay, so we're trading

        use TradingTargetRule::*;
        match self.trading_rule.target {
            Arbitrary => self
                .comp
                .iter()
                .enumerate()
                .filter(|(_ix, &tcls)| tcls == cls)
                .nth(0)
                .map(|(ix, _)| ix),
            MostPieces => {
                let mut target = None;
                let mut target_items = None;
                for (ix, &tcls) in self.comp.iter().enumerate() {
                    if tcls != cls {
                        continue;
                    }
                    let items = self.num_slots[ix];
                    if items < 4 && items > target_items.unwrap_or(0) {
                        target = Some(ix);
                        target_items = Some(items);
                    }
                }

                target
            }
            LeastPieces => {
                // FIXME: copypasta
                let mut target = None;
                let mut target_items = None;
                for (ix, &tcls) in self.comp.iter().enumerate() {
                    if tcls != cls {
                        continue;
                    }
                    let items = self.num_slots[ix];
                    if items < target_items.unwrap_or(5) {
                        target = Some(ix);
                        target_items = Some(items);
                    }
                }

                target
            }
        }
    }

    fn award_tier<R: Rng>(&mut self, rng: &mut R, slot: Slot) {
        let n = self.num_items(rng);
        let awardees = self
            .comp
            .iter()
            .cloned()
            .enumerate()
            .choose_multiple(rng, n);

        for (ix, cls) in awardees {
            if let Some(other) = self.trade_item(ix, slot, cls) {
                self.store_tier(other, slot);
            } else {
                self.store_tier(ix, slot);
            }
        }
    }

    fn completion_pct(&self, bonus: u8) -> f64 {
        let n_players = self.comp.len() as f64;
        self.num_slots.iter().filter(|&&c| c >= bonus).count() as f64 / n_players
    }

    fn from_settings(settings: &Settings) -> State {
        let comp = settings.comp.members.values().cloned().collect::<Vec<_>>();
        let num_players = comp.len();
        let slots = bitarr![0; 30];

        State {
            comp,
            bonus_chance: Bernoulli::from_ratio((num_players % 5) as u32, 5).unwrap(),
            has: SLOTS
                .iter()
                .cloned()
                .map(|slot| (slot, slots.clone()))
                .collect(),
            num_slots: vec![0; num_players],
            trading_rule: settings.trading_rule.clone(),
            nonraid_vault_chance: Binomial::new(0.2, settings.num_extra_vault_items as u64).unwrap(),
        }
    }

    /// Calculate vault drops for each raider. Slightly inaccurate first week because it doesn't
    /// model the number of items that you can't get from the final 3 on raid row, only prevents you
    /// from getting inappropriate tier slots.
    ///
    /// Also assumes you can get those slots from M+/PvP rows first week, because why would
    /// blizzard be consistent.
    fn award_vault<R: Rng>(&mut self, rng: &mut R, full_raid: bool) {
        /// number of distinct items you can get from raid
        const RAID_ITEMS: f64 = 42.0;

        let raid_dist = Binomial::new(5.0 / RAID_ITEMS, 3).unwrap();
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
                if !self.has[&slot][ix] {
                    self.store_tier(ix, slot);
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct Sample {
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
        for _ in 0..settings.clears_per_week {
            state.award_tier(&mut rng, Slot::Helm);
            state.award_tier(&mut rng, Slot::Legs);
            state.award_tier(&mut rng, Slot::Gloves);
            if weeks > 1 {
                state.award_tier(&mut rng, Slot::Shoulders);
                state.award_tier(&mut rng, Slot::Chest);
            }
        }

        state.award_vault(&mut rng, weeks > 1);
        pct_2pc.push(state.completion_pct(2));

        let completion = state.completion_pct(4);
        pct_4pc.push(completion);

        if completion >= 1.0 {
            break;
        }
    }

    Sample {
        n_weeks: weeks,
        pct_2pc,
        pct_4pc,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::from_args();
    let settings_raw = std::fs::read(&opts.config)?;
    let settings = toml::from_slice::<Settings>(&settings_raw)?;

    let mut samples = Vec::with_capacity(settings.num_samples);

    (0..settings.num_samples)
        .into_par_iter()
        .map(|_ix| sample_completion_time(&settings))
        .collect_into_vec(&mut samples);

    std::fs::write(&opts.output, serde_json::to_vec(&samples)?)?;

    let weeks = samples.iter().map(|s| s.n_weeks as f64).collect::<Vec<_>>();

    let avg_weeks = weeks.iter().mean();
    let var_weeks = weeks.variance();

    println!("Avg weeks: {avg_weeks} (var: {var_weeks})");

    Ok(())
}
