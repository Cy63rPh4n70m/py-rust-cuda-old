use rand::Rng;

pub fn gen_random_name(length: u64) -> String
{
    let all_chars: String = String::from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_-");

    let mut random_name: String = String::new();

    let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
    for _ in 0..length
    {
        let random_idx: usize = rand_gen.gen_range(0..all_chars.len());
        let random_char: char = all_chars.chars().nth(random_idx).unwrap();
        random_name.push(random_char);
    }

    return random_name;
}