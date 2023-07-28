@click.command()
@click.argument('task', type=str)
@click.option('--repeat', type=int, default=100)
@click.option('--avg-worker', type=int, default=0,
              help='whether to use the average worker (== 0), a better worker (< 0) or a worse worker (> 0)')
@click.option('--verbose', type=int, default=0)
@click.option('--no-save', is_flag=True, default=False)
def main(task, repeat, avg_worker, verbose, no_save):
    args = {
        'task': task,
        'repeat': repeat,
        'avg_worker': avg_worker,
        'verbose': verbose,
        'save_results': not no_save
    }
    returns, policy_params, avg_return, std_return, cv, worker = test_task(args)
    print_results(returns, policy_params, avg_return, std_return, cv, args['task'],
                  save_results=not no_save, worker=worker)


if __name__ == "__main__":
    main()