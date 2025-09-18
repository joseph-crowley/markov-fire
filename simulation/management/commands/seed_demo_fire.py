from django.core.management.base import BaseCommand

from simulation.services.demo import generate_demo_run


class Command(BaseCommand):
    help = "Seed the demo fire corridor scenario and generate a deterministic run."

    def add_arguments(self, parser):
        parser.add_argument('--reset', action='store_true', help='Recreate scenario, versions, and demo run.')

    def handle(self, *args, **options):
        result = generate_demo_run(reset=options['reset'])
        run = result.run
        self.stdout.write(self.style.SUCCESS(f'Demo run created with {run.total_ticks} ticks.'))
        self.stdout.write(self.style.MIGRATE_HEADING('Demo scenario and run ready.'))
